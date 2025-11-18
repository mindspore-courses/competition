# Copyright (c) Tencent Inc. All rights reserved.
from .conv import Conv

import mindspore.nn as nn
from mindspore import Tensor
import mindspore as ms

__all__ = ('MaxSigmoidAttnBlock')


class MaxSigmoidAttnBlock(nn.Cell):
    """Max Sigmoid attention block."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 guide_channels: int,
                 embed_channels: int,
                 kernel_size: int = 3,
                 padding: int = 1,
                 num_heads: int = 1,
                 with_scale: bool = False,
                 with_norm: bool = True,
                 use_einsum: bool = True) -> None:
        super().__init__()

        assert (out_channels % num_heads == 0 and
                embed_channels % num_heads == 0), \
            'out_channels and embed_channels should be divisible by num_heads.'
        self.num_heads = num_heads
        self.head_channels = out_channels // num_heads
        self.use_einsum = use_einsum

        self.embed_conv = Conv(
            in_channels, embed_channels, 1, with_norm=with_norm,
            with_activation=False) if embed_channels != in_channels else None

        self.guide_fc = nn.Dense(guide_channels, embed_channels)
        self.bias = ms.Parameter(ms.ops.zeros(num_heads))
        if with_scale:
            self.scale = ms.Parameter(ms.ops.ones(1, num_heads, 1, 1))
        else:
            self.scale = 1.0

        self.project_conv = Conv(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=padding,
            with_norm=with_norm,
            with_activation=False)

    def construct(self, x: Tensor, guide: Tensor) -> Tensor:
        B, _, H, W = x.shape

        guide = self.guide_fc(guide)
        # guide = guide.reshape(B, -1, self.num_heads, self.head_channels)
        guide = ms.ops.reshape(guide, (B, -1, self.num_heads, self.head_channels))
        embed = self.embed_conv(x) if self.embed_conv is not None else x
        # embed = embed.reshape(B, self.num_heads, self.head_channels, H, W)
        embed = ms.ops.reshape(embed, (B, self.num_heads, self.head_channels, H, W))

        if self.use_einsum:
            # attn_weight = torch.einsum('bmchw,bnmc->bmhwn', embed, guide)
            attn_weight = ms.ops.einsum('bmchw,bnmc->bmhwn', embed, guide)
        else:
            batch, m, channel, height, width = embed.shape
            _, n, _, _ = guide.shape
            embed = embed.permute(0, 1, 3, 4, 2)
            embed = embed.reshape(batch, m, -1, channel)
            guide = guide.permute(0, 2, 3, 1)
            # attn_weight = torch.matmul(embed, guide)
            attn_weight = ms.ops.matmul(embed, guide)
            attn_weight = attn_weight.reshape(batch, m, height, width, n)

        attn_weight = attn_weight.max(axis=-1)[0]
        attn_weight = attn_weight / (self.head_channels**0.5)
        attn_weight = attn_weight + self.bias[None, :, None, None]
        attn_weight = ms.ops.sigmoid(attn_weight) * self.scale

        x = self.project_conv(x)
        # x = x.reshape(B, self.num_heads, -1, H, W)
        x = ms.ops.reshape(x, (B, self.num_heads, -1, H, W))
        # x = x * attn_weight.unsqueeze(2)
        x = x * ms.ops.unsqueeze(attn_weight, 2)
        # x = x.reshape(B, -1, H, W)
        x = ms.ops.reshape(x, (B, -1, H, W))
        return x