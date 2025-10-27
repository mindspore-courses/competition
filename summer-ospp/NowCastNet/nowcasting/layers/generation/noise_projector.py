import mindspore.nn as nn
import mindspore.ops as ops
from ..utils import SpectralNormal

class NoiseProjector(nn.Cell):
    """Noise Projector"""
    def __init__(self, t_in):
        super(NoiseProjector, self).__init__()
        self.conv_first = SpectralNormal(nn.Conv2d(t_in,
                                                   t_in * 2,
                                                   kernel_size=3,
                                                   pad_mode='pad',
                                                   padding=1,
                                                   has_bias=True
                                                   ),
                                         )
        self.block1 = ProjBlock(t_in * 2, t_in * 4)
        self.block2 = ProjBlock(t_in * 4, t_in * 8)
        self.block3 = ProjBlock(t_in * 8, t_in * 16)
        self.block4 = ProjBlock(t_in * 16, t_in * 32)

    def construct(self, x):
        x = self.conv_first(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        out = self.block4(x)
        return out


class ProjBlock(nn.Cell):
    """Projector block"""
    def __init__(self, in_channels, out_channels):
        super(ProjBlock, self).__init__()
        self.one_conv = SpectralNormal(nn.Conv2d(in_channels, out_channels - in_channels,
                                                 kernel_size=1, has_bias=True))
        self.double_conv = nn.SequentialCell(
            SpectralNormal(nn.Conv2d(in_channels, out_channels, kernel_size=3, pad_mode='pad',
                                     padding=1, has_bias=True),),
            nn.ReLU(),
            SpectralNormal(nn.Conv2d(out_channels, out_channels, kernel_size=3, pad_mode='pad',
                                     padding=1, has_bias=True))
        )

    def construct(self, x):
        x1 = ops.concat([x, self.one_conv(x)], axis=1)
        x2 = self.double_conv(x)
        out = x1 + x2
        return out