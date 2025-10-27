import mindspore.nn as nn
import mindspore.ops as ops
from ..utils import SpectralNormal

class ReflectPad(nn.Cell):
    def __init__(self, padding):
        super(ReflectPad, self).__init__()
        self.pad = nn.Pad(paddings=((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="REFLECT")

    def construct(self, x): 
        return self.pad(x)

class GenBlock(nn.Cell):
    """GenBlock"""
    def __init__(self, in_channels, out_channels, config, dilation=1, double_conv=False):
        super(GenBlock, self).__init__()
        self.learned_shortcut = (in_channels != out_channels)
        self.double_conv = double_conv
        mid_channels = min(in_channels, out_channels)
        t_out = config.total_length - config.input_length
        self.pad = ReflectPad(dilation)
        self.conv_0 = nn.Conv2d(in_channels,
                                mid_channels,
                                kernel_size=3,
                                pad_mode='valid',
                                has_bias=True,
                                dilation=dilation
                                )
        self.conv_0 = SpectralNormal(self.conv_0)
        self.norm_0 = SPADE(in_channels, t_out)
        self.conv_1 = nn.Conv2d(mid_channels,
                                out_channels,
                                kernel_size=3,
                                pad_mode='valid',
                                has_bias=True,
                                dilation=dilation
                                )
        self.conv_1 = SpectralNormal(self.conv_1)
        self.norm_1 = SPADE(mid_channels, t_out)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(in_channels, out_channels, kernel_size=1, pad_mode='pad')
            self.conv_s = SpectralNormal(self.conv_s)
            self.norm_s = SPADE(in_channels, t_out)
        self.leaky_relu = nn.LeakyReLU(2e-1)

    def construct(self, x, evo):
        x_s = self.shortcut(x, evo)
        dx = self.conv_0(self.pad(self.leaky_relu(self.norm_0(x, evo))))
        if self.double_conv:
            dx = self.conv_1(self.pad(self.leaky_relu(self.norm_1(dx, evo))))
        out = x_s + dx
        return out

    def shortcut(self, x, evo):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, evo))
        else:
            x_s = x
        return x_s

class SPADE(nn.Cell):
    """SPADE class"""
    def __init__(self, norm_channels, label_nc, hidden=64, kernel_size=3):
        super(SPADE, self).__init__()
        self.param_free_norm = nn.BatchNorm2d(norm_channels, affine=False)
        self.pad_head = ReflectPad(kernel_size // 2)
        self.mlp_shared = nn.SequentialCell(
            nn.Conv2d(label_nc, hidden, kernel_size=kernel_size, pad_mode='pad', has_bias=True),
            nn.ReLU()
        )
        self.pad = ReflectPad(kernel_size // 2)
        self.mlp_gamma = nn.Conv2d(hidden, norm_channels, kernel_size=kernel_size, pad_mode='pad', has_bias=True)
        self.mlp_beta = nn.Conv2d(hidden, norm_channels, kernel_size=kernel_size, pad_mode='pad', has_bias=True)


    def construct(self, x, evo):
        normalized = self.param_free_norm(x)
        evo = ops.adaptive_avg_pool2d(evo, output_size=x.shape[2:])
        evo = self.pad_head(evo)
        evo_out = self.mlp_shared(evo)
        gamma = self.mlp_gamma(self.pad(evo_out))
        beta = self.mlp_beta(self.pad(evo_out))
        out = normalized * (1 + gamma) + beta
        return out