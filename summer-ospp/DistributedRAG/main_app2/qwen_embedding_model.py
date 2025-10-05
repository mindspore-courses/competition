# 原生mindspore
import os
import json
import math
import numpy as np
from typing import List, Union
from dataclasses import dataclass
import dataclasses

import torch
import mindspore as ms
from mindspore import nn, ops, Tensor, Parameter
from mindspore.train.serialization import load_param_into_net
from safetensors.torch import load_file
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download


@dataclass
class QwenConfig:
    hidden_size: int = 1024
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    intermediate_size: int = 3072
    num_hidden_layers: int = 28
    vocab_size: int = 151669
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    max_position_embeddings: int = 32768
    head_dim: int = 128

    @classmethod
    def from_json(cls, json_path: str) -> 'QwenConfig':
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        valid_keys = {f.name for f in dataclasses.fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)

class RmsNorm(nn.Cell):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = Parameter(ops.ones(dim, ms.float32), name="weight")
    def construct(self, x: Tensor):
        variance = x.to(ms.float32).square().mean(-1, keep_dims=True)
        x_norm = x * ops.rsqrt(variance + self.eps)
        return (self.weight * x_norm).to(x.dtype)

class RotaryEmbedding(nn.Cell):
    def __init__(self, head_size: int, max_pos: int, base: int):
        super().__init__()
        inv_freq = 1.0 / (base ** (np.arange(0, head_size, 2, dtype=np.float32) / head_size))
        t = np.arange(max_pos, dtype=np.float32)
        freqs = np.outer(t, inv_freq)
        emb = np.concatenate((freqs, freqs), axis=1)
        self.freqs_cos = Tensor(np.cos(emb), ms.float32)
        self.freqs_sin = Tensor(np.sin(emb), ms.float32)
    def _apply_rotary(self, x, cos, sin):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        rotated = ops.concat((-x2, x1), axis=-1)
        return x * cos + rotated * sin
    def construct(self, pos: Tensor, query: Tensor, key: Tensor):
        cos = ops.expand_dims(ops.expand_dims(self.freqs_cos[pos], 0), 2)
        sin = ops.expand_dims(ops.expand_dims(self.freqs_sin[pos], 0), 2)
        return self._apply_rotary(query, cos, sin), self._apply_rotary(key, cos, sin)

class Attention(nn.Cell):
    def __init__(self, config: QwenConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        q_proj_size = self.num_heads * self.head_dim
        kv_proj_size = self.num_kv_heads * self.head_dim
        self.q_proj = nn.Dense(config.hidden_size, q_proj_size, has_bias=False)
        self.k_proj = nn.Dense(config.hidden_size, kv_proj_size, has_bias=False)
        self.v_proj = nn.Dense(config.hidden_size, kv_proj_size, has_bias=False)
        self.o_proj = nn.Dense(q_proj_size, config.hidden_size, has_bias=False)
        self.q_norm = RmsNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RmsNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(self.head_dim, config.max_position_embeddings, int(config.rope_theta))
    def construct(self, hidden_state, positions, attention_mask):
        bsz, seq_len, _ = hidden_state.shape
        q = self.q_proj(hidden_state).view(bsz, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_state).view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_state).view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        q, k = self.q_norm(q), self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        q, k, v = q.transpose((0, 2, 1, 3)), k.transpose((0, 2, 1, 3)), v.transpose((0, 2, 1, 3))
        if self.num_kv_heads != self.num_heads:
            k = ops.repeat_interleave(k, self.num_heads // self.num_kv_heads, axis=1)
            v = ops.repeat_interleave(v, self.num_heads // self.num_kv_heads, axis=1)
        scores = ops.matmul(q.to(ms.float32), k.to(ms.float32).transpose((0, 1, 3, 2))) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            scores += attention_mask
        probs = ops.softmax(scores, axis=-1).to(hidden_state.dtype)
        attn_out = ops.matmul(probs, v).transpose((0, 2, 1, 3)).reshape(bsz, seq_len, -1)
        return self.o_proj(attn_out)

class MLP(nn.Cell):
    def __init__(self, config: QwenConfig):
        super().__init__()
        self.gate_proj = nn.Dense(config.hidden_size, config.intermediate_size, has_bias=False)
        self.up_proj = nn.Dense(config.hidden_size, config.intermediate_size, has_bias=False)
        self.down_proj = nn.Dense(config.intermediate_size, config.hidden_size, has_bias=False)
    def construct(self, x):
        return self.down_proj(ops.silu(self.gate_proj(x)) * self.up_proj(x))

class DecoderLayer(nn.Cell):
    def __init__(self, config: QwenConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = MLP(config)
        self.input_layernorm = RmsNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RmsNorm(config.hidden_size, eps=config.rms_norm_eps)
    def construct(self, hidden_state, positions, attention_mask):
        residual = hidden_state
        h = self.input_layernorm(hidden_state)
        h = self.self_attn(h, positions, attention_mask)
        h = residual + h
        residual = h
        h = self.post_attention_layernorm(h)
        h = self.mlp(h)
        return residual + h

class QwenTransformer(nn.Cell):
    def __init__(self, config: QwenConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.CellList([DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RmsNorm(config.hidden_size, eps=config.rms_norm_eps)
    def construct(self, input_ids: Tensor, positions: Tensor, attention_mask: Tensor):
        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h = layer(h, positions, attention_mask)
        return self.norm(h)

class EmbeddingModel(nn.Cell):
    def __init__(self, config: QwenConfig):
        super().__init__()
        self.transformer = QwenTransformer(config)
    def construct(self, input_ids: Tensor, attention_mask: Tensor):
        seq_len = input_ids.shape[1]
        positions = ops.arange(seq_len, dtype=ms.int64)
        causal_mask = ops.expand_dims(ops.expand_dims(attention_mask, 1), 2)
        causal_mask = (1.0 - causal_mask) * Tensor(np.finfo(np.float32).min, ms.float32)
        last_hidden_state = self.transformer(input_ids, positions, causal_mask)
        masked_hidden_state = last_hidden_state * ops.expand_dims(attention_mask, -1)
        summed_hidden_state = ops.sum(masked_hidden_state, 1)
        summed_mask = ops.sum(attention_mask, 1, keepdim=True)
        pooled_embedding = summed_hidden_state / ops.maximum(summed_mask, Tensor(1e-9, ms.float32))
        normalized_embedding = ops.L2Normalize(axis=-1)(pooled_embedding)
        return normalized_embedding


class QwenEmbeddingModel:
    def __init__(self, model_name_or_path: str, device: str = "auto"):
        self._set_device(device)
        self.local_path = self._get_local_path(model_name_or_path)
        self.config = QwenConfig.from_json(os.path.join(self.local_path, "config.json"))
        self.tokenizer = AutoTokenizer.from_pretrained(self.local_path, trust_remote_code=True)
        
        # 动态对齐 vocab size
        if self.config.vocab_size != len(self.tokenizer):
            print(f"⚠️ Vocab size mismatch! Updating config from {self.config.vocab_size} to {len(self.tokenizer)}")
            self.config.vocab_size = len(self.tokenizer)
        
        self.model = self._initialize_and_load_weights()
        self.model.set_train(False)

    def _set_device(self, device: str):
        if device.lower() == "auto":
            try:
                ms.set_context(mode=ms.PYNATIVE_MODE, device_target="GPU")
                print("Using device: GPU")
            except Exception:
                ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
                print("Using device: CPU")
        else:
            ms.set_context(mode=ms.PYNATIVE_MODE, device_target=device.upper())
            print(f"Using device: {device.upper()}")
            
    def _get_local_path(self, model_path: str) -> str:
        if os.path.isdir(model_path):
            return model_path
        try:
            return snapshot_download(repo_id=model_path)
        except Exception as e:
            raise IOError(f"Failed to download or find model at '{model_path}'. Error: {e}")

    def _initialize_and_load_weights(self) -> EmbeddingModel:
        model = EmbeddingModel(self.config)
        
        renamed_param_dict = {}
        for f in os.listdir(self.local_path):
            if f.endswith(".safetensors"):
                torch_dict = load_file(os.path.join(self.local_path, f), device="cpu")
                for name, pt_tensor in torch_dict.items():
                    new_name = "transformer." + name
                    if new_name == "transformer.embed_tokens.weight":
                        new_name = "transformer.embed_tokens.embedding_table"
                    renamed_param_dict[new_name] = Parameter(Tensor(pt_tensor.to(torch.float32).numpy()), name=new_name)
        
        load_param_into_net(model, renamed_param_dict, strict_load=False)
        model.to_float(ms.float32)
        return model

    def encode(self, sentences: Union[str, List[str]], max_length: int = 512) -> np.ndarray:
        """
        Args:
            sentences (Union[str, List[str]]): 单个句子或句子列表。
            max_length (int, optional): Tokenizer的最大序列长度。 Defaults to 512.

        Returns:
            np.ndarray: 返回 embedding 向量的 NumPy 数组。
                        如果输入是单个句子，返回 (embedding_dim,) 的向量。
                        如果输入是句子列表，返回 (num_sentences, embedding_dim) 的矩阵。
        """
        is_single_sentence = isinstance(sentences, str)
        if is_single_sentence:
            sentences = [sentences]

        inputs = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="np", max_length=max_length)
        input_ids = Tensor(inputs['input_ids'])
        attention_mask = Tensor(inputs['attention_mask'])
        
        embeddings = self.model(input_ids, attention_mask)
        
        numpy_embeddings = embeddings.asnumpy()
        return numpy_embeddings[0] if is_single_sentence else numpy_embeddings
