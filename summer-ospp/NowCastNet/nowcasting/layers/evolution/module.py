import mindspore.nn as nn
import mindspore.ops as ops
from ..utils import SpectralNormal

class DoubleConv(nn.Cell):
    """Double convolution block with residual connection"""
    def __init__(self, in_channels, out_channels, kernel=3, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.SequentialCell(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            SpectralNormal(nn.Conv2d(in_channels, 
                                    mid_channels, 
                                    kernel_size=kernel, 
                                    padding=kernel//2, 
                                    pad_mode='pad',
                                    has_bias=True
                                    )
                        ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            SpectralNormal(nn.Conv2d(mid_channels, 
                                    out_channels, 
                                    kernel_size=kernel, 
                                    padding=kernel//2, 
                                    pad_mode='pad',
                                    has_bias=True
                                    )
                        ),
        )
        self.single_conv = nn.SequentialCell(
            nn.BatchNorm2d(in_channels),
            SpectralNormal(nn.Conv2d(in_channels, 
                                    out_channels, 
                                    kernel_size=kernel, 
                                    padding=kernel//2, 
                                    pad_mode='pad', 
                                    has_bias=True
                                    )
                        )
        )

    def construct(self, x):
        shortcut = self.single_conv(x)
        x = self.double_conv(x)
        x = x + shortcut
        return x

class Down(nn.Cell):
    """Down sample"""
    def __init__(self, in_channels, out_channels, kernel=3):
        super().__init__()
        self.maxpool_conv = nn.SequentialCell(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels, kernel)
        )

    def construct(self, x):
        out = self.maxpool_conv(x)
        return out

class Up(nn.Cell):
    """Up sample"""
    def __init__(self, in_channels, out_channels, size, bilinear=True, kernel=3):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(size=size, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, kernel=kernel, mid_channels=in_channels // 2)
        else:
            self.up = nn.Conv2dTranspose(in_channels,
                                         in_channels // 2,
                                         kernel_size=2,
                                         stride=2,
                                         has_bias=True,
                                         pad_mode="pad"
                                         )
            self.conv = DoubleConv(in_channels, out_channels, kernel)

    def construct(self, x1, x2):
        x1 = self.up(x1)
        x = ops.cat([x2, x1], axis=1)
        return self.conv(x)

class OutConv(nn.Cell):
    """Output convolution layer"""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, pad_mode='pad', has_bias=True)

    def construct(self, x):
        return self.conv(x) 