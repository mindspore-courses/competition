# 原生mindspore
import os
import json
import math
import numpy as np
from typing import List, Dict
from dataclasses import dataclass
import dataclasses

import mindspore as ms
from mindspore import nn, ops, Tensor, Parameter
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from transformers import AutoTokenizer

@dataclass
class QwenConfig:
    hidden_size: int = 1536
    num_attention_heads: int = 12
    num_key_value_heads: int = 2
    intermediate_size: int = 8960
    num_hidden_layers: int = 28
    vocab_size: int = 151936
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    max_position_embeddings: int = 32768
    tie_word_embeddings: bool = True

    @classmethod
    def from_json(cls, json_path: str) -> 'QwenConfig':
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        valid_keys = {f.name for f in dataclasses.fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)

@dataclass
class ModelInput:
    input_ids: Tensor
    positions: Tensor
    k_caches: List[Tensor]
    v_caches: List[Tensor]
    attn_mask: Tensor
    cur_len: int

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
        self.head_size = head_size
        inv_freq = 1.0 / (base ** (np.arange(0, head_size, 2, dtype=np.float32) / head_size))
        t = np.arange(max_pos, dtype=np.float32)
        freqs = np.outer(t, inv_freq)
        emb = np.concatenate((freqs, freqs), axis=1)
        self.freqs_cos = Tensor(np.cos(emb), ms.float32)
        self.freqs_sin = Tensor(np.sin(emb), ms.float32)
    def _apply_rotary(self, x, cos, sin):
        x1, x2 = x[..., :self.head_size//2], x[..., self.head_size//2:]
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
        self.head_dim = config.hidden_size // config.num_attention_heads
        q_proj_size = self.num_heads * self.head_dim
        kv_proj_size = self.num_kv_heads * self.head_dim
        self.q_proj = nn.Dense(config.hidden_size, q_proj_size, has_bias=True)
        self.k_proj = nn.Dense(config.hidden_size, kv_proj_size, has_bias=True)
        self.v_proj = nn.Dense(config.hidden_size, kv_proj_size, has_bias=True)
        self.o_proj = nn.Dense(q_proj_size, config.hidden_size, has_bias=False)
        self.rotary_emb = RotaryEmbedding(self.head_dim, config.max_position_embeddings, int(config.rope_theta))
    def construct(self, hidden_state, positions, k_cache, v_cache, attn_mask, cur_len):
        bsz, seq_len, _ = hidden_state.shape
        q = self.q_proj(hidden_state).view(bsz, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_state).view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_state).view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        q, k = self.rotary_emb(positions, q, k)
        k_cache[:, cur_len : cur_len + seq_len] = k
        v_cache[:, cur_len : cur_len + seq_len] = v
        k_all, v_all = k_cache[:, : cur_len + seq_len], v_cache[:, : cur_len + seq_len]
        q, k, v = q.transpose((0, 2, 1, 3)), k_all.transpose((0, 2, 1, 3)), v_all.transpose((0, 2, 1, 3))
        if self.num_kv_heads != self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            k = ops.repeat_interleave(k, repeat, axis=1)
            v = ops.repeat_interleave(v, repeat, axis=1)
        scores = ops.matmul(q, k.transpose((0, 1, 3, 2))) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            scores += attn_mask
        probs = ops.softmax(scores.astype(ms.float32), axis=-1).astype(hidden_state.dtype)
        attn_out = ops.matmul(probs, v)
        attn_out = attn_out.transpose((0, 2, 1, 3)).reshape(bsz, seq_len, -1)
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
    def construct(self, hidden_state, positions, k_cache, v_cache, attn_mask, cur_len):
        residual = hidden_state
        h = self.input_layernorm(hidden_state)
        h = self.self_attn(h, positions, k_cache, v_cache, attn_mask, cur_len)
        h = residual + h
        residual = h
        h = self.post_attention_layernorm(h)
        h = self.mlp(h)
        return residual + h

class Model(nn.Cell):
    def __init__(self, config: QwenConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.CellList([DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RmsNorm(config.hidden_size, eps=config.rms_norm_eps)
    def construct(self, input_ids, positions, k_caches, v_caches, attn_mask, cur_len):
        h = self.embed_tokens(input_ids)
        for i, layer in enumerate(self.layers):
            h = layer(h, positions, k_caches[i], v_caches[i], attn_mask, cur_len)
        return self.norm(h)

class ForCausalLM(nn.Cell):
    def __init__(self, config: QwenConfig):
        super().__init__()
        self.model = Model(config)
    def construct(self, input_obj: ModelInput):
        h = self.model(input_obj.input_ids, input_obj.positions,
                       input_obj.k_caches, input_obj.v_caches,
                       input_obj.attn_mask, input_obj.cur_len)
        return ops.matmul(h[:, -1, :], self.model.embed_tokens.embedding_table.T)

class CacheManager:
    def __init__(self, config: QwenConfig, max_seq_len: int, batch_size: int, dtype=ms.float16):
        head_dim = config.hidden_size // config.num_attention_heads
        self.k_caches = ms.mutable([
            ops.zeros((batch_size, max_seq_len, config.num_key_value_heads, head_dim), dtype)
            for _ in range(config.num_hidden_layers)
        ])
        self.v_caches = ms.mutable([
            ops.zeros((batch_size, max_seq_len, config.num_key_value_heads, head_dim), dtype)
            for _ in range(config.num_hidden_layers)
        ])


class QwenCausalLM:
    """
    An interface for running inference with the Qwen Causal Language Model using MindSpore.
    """
    def __init__(self, model_path: str, device: str = "auto"):
        self._set_device(device)
        print(f"Loading model from path: {model_path}")
        if not os.path.isdir(model_path):
            raise NotADirectoryError(f"The provided model path is not a valid directory: {model_path}")

        config_path = os.path.join(model_path, "config.json")
        self.config = QwenConfig.from_json(config_path)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        self.model = self._initialize_and_load_weights(model_path)
        print("✅ Model is ready for inference.")

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

    def _initialize_and_load_weights(self, model_path: str) -> ForCausalLM:
        """Initializes the model and loads weights from checkpoint, handling key renaming."""
        model = ForCausalLM(self.config)
        
        ckpt_path = os.path.join(model_path, "mindspore_model.ckpt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"MindSpore checkpoint not found at: {ckpt_path}")
            
        param_dict = load_checkpoint(ckpt_path)

        # --- Handle complex key renaming ---
        # Rename embedding key
        target_key = "model.embed_tokens.embedding_table"
        for candidate in ["model.embed_tokens.weight", "transformer.wte.weight", target_key, "wte.weight", "embed_tokens.weight"]:
            if candidate in param_dict and candidate != target_key:
                param_dict[target_key] = param_dict.pop(candidate)
                break

        # Rename layer keys based on common prefixes
        prefix_mapping = {
            "transformer.h.": "model.layers.",
            "transformer.ln_f.weight": "model.norm.weight",
        }
        renamed_dict = {}
        for old_key, value in param_dict.items():
            new_key = old_key
            for old_prefix, new_prefix in prefix_mapping.items():
                if old_key.startswith(old_prefix):
                    new_key = new_prefix + old_key[len(old_prefix):]
                    break
            renamed_dict[new_key] = value
        
        load_param_into_net(model, renamed_dict, strict_load=False)

        # Align vocab size between tokenizer and model embedding table
        actual_vocab_size = len(self.tokenizer)
        model_vocab_size = model.model.embed_tokens.embedding_table.shape[0]
        if model_vocab_size != actual_vocab_size:
            print(f"⚠️ Vocab size mismatch! Resizing model embedding from {model_vocab_size} to {actual_vocab_size}")
            old_weight = model.model.embed_tokens.embedding_table
            new_weight = Parameter(old_weight[:actual_vocab_size], name="embedding_table")
            model.model.embed_tokens.embedding_table = new_weight

        model.to_float(ms.float32)
        model.set_train(False)
        return model

    def generate(self, messages: List[Dict[str, str]], max_new_tokens: int = 256, max_seq_len: int = 4096) -> str:
        """
        Generates a text response based on the input messages.

        Args:
            messages (List[Dict[str, str]]): A list of message dictionaries, e.g., [{"role": "user", "content": "Hello"}].
            max_new_tokens (int): The maximum number of new tokens to generate.
            max_seq_len (int): The maximum sequence length for the model.

        Returns:
            str: The generated text response.
        """
        self.model.set_train(False)
        dtype = ms.float16
        cache_manager = CacheManager(self.config, max_seq_len, 1, dtype=dtype)
        
        input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        
        # Sanitize input token IDs to be within the tokenizer's vocab size
        actual_vocab_size = len(self.tokenizer)
        input_ids = [tid if tid < actual_vocab_size else self.tokenizer.unk_token_id for tid in input_ids]

        generated_tokens = []
        is_prefill = True
        
        while len(generated_tokens) < max_new_tokens:
            cur_len = 0 if is_prefill else len(input_ids) - 1
            inp_list = input_ids if is_prefill else [input_ids[-1]]
            
            inp = Tensor(np.array([inp_list]), ms.int64)
            seq_len_now = inp.shape[1]
            pos = ops.arange(cur_len, cur_len + seq_len_now, dtype=ms.int64)
            
            attn_mask = None
            if seq_len_now > 1:
                mask = ops.ones((seq_len_now, cur_len + seq_len_now), dtype=dtype)
                attn_mask = ops.triu(mask, diagonal=cur_len + 1) * Tensor(np.finfo(np.float16).min, dtype=dtype)
            
            model_input = ModelInput(inp, pos, cache_manager.k_caches, cache_manager.v_caches, attn_mask, cur_len)
            logits = self.model(model_input)
            
            token_val = ops.argmax(logits, dim=-1).item()

            if token_val == self.tokenizer.eos_token_id:
                break

            generated_tokens.append(token_val)
            input_ids.append(token_val)
            is_prefill = False

        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

