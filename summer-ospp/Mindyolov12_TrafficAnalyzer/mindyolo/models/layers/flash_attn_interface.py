"""
Flash Attention Interface (pure MindSpore ops)
"""
import mindspore as ms
from mindspore import ops, Tensor
import mindspore.common.dtype as mstype


def _build_masks(seqlen: int, causal: bool, window_size):
    """Create attention masks in fp16 with large negative value to be added to logits.
    Returns a tensor with shape (1, seqlen, seqlen) or None.
    """
    attn_mask = None
    if causal:
        mask_shape = (1, seqlen, seqlen)
        upper = ops.triu(ops.ones(mask_shape, mstype.float16), diagonal=1)
        attn_mask = upper * (-65504.0)  # ~ -inf in fp16

    if window_size != (-1, -1):
        left_window, right_window = window_size
        if left_window >= 0 or right_window >= 0:
            idx = ops.arange(seqlen)
            row = idx.view(-1, 1)
            col = idx.view(1, -1)
            distance = (col - row).astype(mstype.int32)  # (S, S)

            window_mask = ops.zeros((seqlen, seqlen), mstype.float16)
            if left_window >= 0:
                window_mask = ops.where(
                    distance < Tensor(-int(left_window), mstype.int32),
                    Tensor(-65504.0, mstype.float16),
                    window_mask,
                )
            if right_window >= 0:
                window_mask = ops.where(
                    distance > Tensor(int(right_window), mstype.int32),
                    Tensor(-65504.0, mstype.float16),
                    window_mask,
                )
            window_mask = window_mask.view(1, seqlen, seqlen)
            attn_mask = window_mask if attn_mask is None else (attn_mask + window_mask)

    return attn_mask


def _manual_attention(q, k, v, softmax_scale=None, attn_mask=None):
    """Manual attention with MindSpore ops.
    Inputs are (B, H, S, D). Returns (B, H, S, D).
    """
    B, H, S, D = q.shape
    scale = softmax_scale if softmax_scale is not None else 1.0 / (D ** 0.5)

    # logits: (B, H, S, S)
    logits = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * Tensor(scale, q.dtype)
    if attn_mask is not None:
        # attn_mask: (1 or B, S, S) -> (B, 1, S, S)
        if attn_mask.shape[0] == 1 and B > 1:
            attn_mask = ops.tile(attn_mask, (B, 1, 1))
        logits = logits + attn_mask.view(B, 1, S, S)

    # Numerically stable softmax (MindSpore softmax internally stabilized, extra safety optional)
    attn = ops.softmax(logits, axis=-1)
    out = ops.matmul(attn, v)
    return out


def flash_attn_func(
    q,
    k,
    v,
    dropout_p: float = 0.0,
    softmax_scale=None,
    causal: bool = False,
    window_size=(-1, -1),
    alibi_slopes=None,  # placeholder
    deterministic: bool = False,  # placeholder
):
    """
    FlashAttention-like接口（纯 MindSpore ops 实现）。

    Args:
        q, k, v: shape 可为 (B, S, H, D) 或 (B, H, S, D)
        dropout_p: 仅保留接口，不生效（训练时可在外层加 Dropout）
        softmax_scale: 注意力缩放系数，默认 1/sqrt(D)
        causal: 自回归掩码
        window_size: (left, right) 滑动窗口；-1 表示不限制
        alibi_slopes, deterministic: 仅保留接口
    Returns:
        与 q 相同布局的张量
    """
    if q is None or k is None or v is None:
        raise ValueError("q, k, v must be provided")
    if len(q.shape) != 4:
        raise ValueError(f"Expected 4D input tensors, got shape {q.shape}")

    original_dtype = q.dtype

    # 统一为 (B, H, S, D)
    need_transpose_back = False
    # 经验判断：若第二维 > 第三维，通常为 (B, S, H, D)
    if q.shape[1] > q.shape[2]:
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        need_transpose_back = True

    B, H, S, D = q.shape

    # Ascend 上建议用 fp16 计算，计算后还原 dtype
    calc_dtype = mstype.float16 if original_dtype not in (mstype.float16, mstype.float32) else original_dtype
    q = q.astype(calc_dtype)
    k = k.astype(calc_dtype)
    v = v.astype(calc_dtype)

    attn_mask = _build_masks(S, causal, window_size)

    out = _manual_attention(q, k, v, softmax_scale=softmax_scale, attn_mask=attn_mask)
    out = out.astype(original_dtype)

    if need_transpose_back:
        out = out.transpose(0, 2, 1, 3)
    return out


def flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    block_table=None,
):
    """
    变长序列的简化实现：直接调用定长实现（上游如需真变长优化，可在外层对齐填充后调用）。
    """
    return flash_attn_func(
        q,
        k,
        v,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
    )


def flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens=None,
    cache_batch_idx=None,
    block_table=None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    rotary_interleaved=True,
    alibi_slopes=None,
):
    """
    带 KV 缓存的简化实现：若提供 k/v 则直接计算，否则使用缓存。
    """
    if k is not None and v is not None:
        return flash_attn_func(
            q,
            k,
            v,
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=False,
        )
    else:
        return flash_attn_func(
            q,
            k_cache,
            v_cache,
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=False,
        )