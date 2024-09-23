# Copyright (c) Tencent Inc. All rights reserved.
import mindspore.nn as nn
from typing import Tuple, Union

import mindspore as ms
from mindspore import Tensor

__all__ = ('Conv')

class Conv(nn.Cell):
    """A convolution block
    composed of conv/norm/activation layers.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1,
                 with_norm: bool = False,
                 with_activation: bool = True,
                 bias: Union[bool, str] = 'auto'):
        super().__init__()
        self.with_norm = with_norm
        self.with_activation = with_activation
        # if the conv layer is before a norm layer, bias is unnecessary.
        if bias == 'auto':
            bias = not self.with_norm
        self.with_bias = bias

        # pytorch与mindspore 的 nn.Conv2d 区别
        # build convolution layer
        if padding > 0:
            pad_mode = 'pad'
        else:
            pad_mode = 'valid'
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            pad_mode=pad_mode,
            padding=padding,
            dilation=dilation,
            group=groups,
            has_bias=self.with_bias,
            )
        
        

        # build normalization layers
        if self.with_norm:
            self.bn = nn.BatchNorm2d(out_channels, momentum=1-0.03, eps=0.001)  # 官方说 pytorch这里的BN中momentum_torch = 1 - momentum_mindspore 
        # build activation layer
        if self.with_activation:
            self.activate = nn.SiLU() # mindspore has no inplace

        # self.init_weights()

    def construct(self, x: Tensor) -> Tensor:
        # fixed order: ('conv', 'norm', 'act')
        x = self.conv(x)
        if self.with_norm:
            x = self.bn(x)
        if self.with_activation:
            x = self.activate(x)
        return x

    # 推理按理说应该不看重init
    def init_weights(self):
        from mindspore.common.initializer import initializer, HeNormal
        # nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        self.conv.weight.set_data(initializer(
                                                HeNormal(negative_slope=0, mode='fan_out', nonlinearity='relu'),
                                                self.conv.weight.shape, self.conv.weight.dtype))
        if hasattr(self.conv, 'bias') and self.conv.bias is not None:
            # nn.init.constant_(self.conv.bias, 0)
            self.conv.bias.set_data(initializer("zeros", self.conv.bias.shape, self.conv.bias.dtype))
            
        
        if self.with_norm:
            # nn.init.constant_(self.bn.weight, 1)
            self.bn.weight.set_data(initializer("ones", self.bn.weight.shape, self.bn.weight.dtype))
            
            if hasattr(self.conv, 'bias') and self.conv.bias is not None:
                # nn.init.constant_(self.bn.bias, 0)
                self.bn.bias.set_data(initializer("zeros", self.bn.bias.shape, self.bn.bias.dtype))