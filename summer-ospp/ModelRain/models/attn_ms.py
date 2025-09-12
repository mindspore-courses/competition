import mindspore.nn as nn
import mindspore as ms
import mindspore.ops as ops
import numpy as np

from utils.masking_ms import TriangularCausalMask, ProbMask, ProbMaskCell
from utils.tools_ms import mask_fill

class FullAttention(nn.Cell):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False, args=None):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(p=attention_dropout)

    def construct(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape

        scores = ops.BatchMatMul()(queries.transpose(0, 2, 1, 3), keys.transpose(0, 2, 3, 1))
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L)
            scores = mask_fill(attn_mask.mask, scores, -np.inf)
        if self.scale is None:
            A = self.dropout(ops.Softmax()((1./ops.sqrt(ms.Tensor(E, ms.float32))) * scores))
        else:
            A = self.dropout(ops.Softmax()(self.scale * scores))
        
        # A = self.dropout(ops.Softmax()(self.scale * scores))
        V = ops.BatchMatMul()(A, values.transpose((0, 2, 1, 3))).transpose(0, 2, 1, 3)
        return V

from mindspore.ops.function import broadcast_to
from mindspore import numpy as ms_np
import mindspore as ms

class ProbAttention(nn.Cell):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False, args=None):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(p=attention_dropout)
        self.device = args.device

    def _prob_QK(self, Q, K, sample_k, n_top): # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        index_sample = ops.UniformInt()((L_Q, sample_k), ms.Tensor(0, ms.int32), ms.Tensor(L_K, ms.int32)) # real U = U_part(factor*ln(L_k))*L_q
        index_sample[index_sample == L_K] = L_K - 1
        
        if self.device == "Ascend":
            K_samples = []
            for i in range(L_Q):
                K_samples.append(ops.gather(K, index_sample[i], axis=2).unsqueeze(2))
            K_sample = ops.cat(K_samples, axis=2)
        else:
            K_expand = ops.BroadcastTo(shape = (B, H, L_Q, L_K, E))(ops.expand_dims(K, -3))
            K_sample = K_expand[:, :, ops.expand_dims(ms.numpy.arange(L_Q), 1), index_sample, :]
        
        Q_K_sample = ops.Squeeze(-2)(ops.BatchMatMul()(ops.expand_dims(Q, -2), K_sample.swapaxes(-2, -1)))
        # find the Top_k query with sparisty measurement
        M = ops.ArgMaxWithValue(-1)(Q_K_sample)[0] - ops.div(ops.ReduceSum()(Q_K_sample, -1), L_K)

        M_top = ops.TopK(sorted=False)(M, n_top)[1]
         # use the reduced Q to calculate Q_K
        Q_reduce = Q[ms.numpy.arange(B)[:, None, None],
                     ms.numpy.arange(H)[None, :, None],
                     M_top, :] # factor*ln(L_q)
        Q_K = ops.BatchMatMul()(Q_reduce, K.swapaxes(-2, -1)) # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            V_sum = ops.ReduceMean()(V, -2)
            contex = ops.BroadcastTo(shape = (B, H, L_Q, V_sum.shape[-1]))(ops.expand_dims(V_sum, -2)).copy()
        else: # use mask
            assert(L_Q == L_V) # requires that L_Q == L_V, i.e. for self-attention only
            contex = ops.cumsum(V, -2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape
        if self.mask_flag:
            _mask = ops.ones((L_Q, scores.shape[-1]), ms.bool_).triu(diagonal=1)
            # _mask = ops.triu(ops.ones((L_Q, scores.shape[-1]), ms.bool_), diagonal=1)
            _mask_ex = broadcast_to(_mask[None, None, :], (B, H, L_Q, scores.shape[-1]))
            indicator = _mask_ex[ms_np.arange(B)[:, None, None],
                                ms_np.arange(H)[None, :, None],
                                index, :]
            final_mask = indicator.view(scores.shape)

            scores = mask_fill(final_mask, scores, -np.inf)
            # attn_mask = ProbMask(B, H, L_Q, index, scores)
            # scores = mask_fill(attn_mask.mask, scores, -np.inf)
        attn = ops.Softmax()(scores)
        context_in[ms.numpy.arange(B)[:, None, None],
                   ms.numpy.arange(H)[None, :, None],
                   index, :] = ops.BatchMatMul()(attn, V).astype(context_in.dtype)

        attns = (ops.ones(((B, H, L_V, L_V)), ms.float32) / L_V).astype(attn.dtype)
        attns[ms.numpy.arange(B)[:, None, None], ms.numpy.arange(H)[None, :, None], index, :] = attn
        return context_in

    def construct(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape
        queries = queries.swapaxes(2, 1)
        keys = keys.swapaxes(2, 1)
        values = values.swapaxes(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item() # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item() # c*ln(L_q) 

        U_part = U_part if U_part<L_K else L_K
        u = u if u<L_Q else L_Q
        
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u) 

        # add scale factor
        if self.scale is None:
            scores_top = scores_top * (1./ops.sqrt(ms.Tensor(D, ms.float32)))
        else:
            scores_top = scores_top * self.scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context = self._update_context(context, values, scores_top, index, L_Q, attn_mask)
        
        return context.swapaxes(2, 1)

class AttentionLayer(nn.Cell):
    def __init__(self, attention, d_model, n_heads, 
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Dense(d_model, d_keys * n_heads)
        self.key_projection = nn.Dense(d_model, d_keys * n_heads)
        self.value_projection = nn.Dense(d_model, d_values * n_heads)
        self.out_projection = nn.Dense(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def construct(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
        )
        if self.mix:
            out = out.swapaxes(2, 1)
        out = out.view(B, L, -1)
        return self.out_projection(out)
