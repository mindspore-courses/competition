import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
import numpy as np
from .module import GenBlock
from ..evolution.module import *
from .noise_projector import NoiseProjector

class GenerativeEncoder(nn.Cell):
    """Encoder of Generative"""
    def __init__(self, in_channels, hidden=64):
        super(GenerativeEncoder, self).__init__()
        self.inc = DoubleConv(in_channels, hidden, kernel=3)
        self.down1 = Down(hidden, hidden * 2, 3)
        self.down2 = Down(hidden * 2, hidden * 4, 3)
        self.down3 = Down(hidden * 4, hidden * 8, 3)

    def construct(self, x):
        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        out = self.down3(x)
        return out


class GenerativeDecoder(nn.Cell):
    """Decoder of Generative"""
    def __init__(self, config):
        super(GenerativeDecoder, self).__init__()
        scale = config.noise_scale // 8
        nf = config.ngf
        in_channels = (8 + nf // (scale**2)) * nf
        out_channels = config.total_length - config.input_length
        self.fc = nn.Conv2d(in_channels, 8 * nf, kernel_size=3, pad_mode='pad', padding=1, has_bias=True)
        self.head_0 = GenBlock(8 * nf, 8 * nf, config)
        self.g_middle_0 = GenBlock(8 * nf, 4 * nf, config, double_conv=True)
        self.g_middle_1 = GenBlock(4 * nf, 4 * nf, config, double_conv=True)
        self.up_0 = GenBlock(4 * nf, 2 * nf, config)
        self.up_1 = GenBlock(2 * nf, nf, config, double_conv=True)
        self.up_2 = GenBlock(nf, nf, config, double_conv=True)
        self.conv_img = nn.Conv2d(nf, out_channels, kernel_size=3, pad_mode='pad', padding=1, has_bias=True)
        self.leaky_relu = nn.LeakyReLU(2e-1)

    def construct(self, x, evo):
        """decoder construct"""
        x = self.fc(x)
        x = self.head_0(x, evo)
        h, w = x.shape[2], x.shape[3]
        x = ops.interpolate(x, size=(h * 2, w * 2))
        x = self.g_middle_0(x, evo)
        x = self.g_middle_1(x, evo)
        h, w = x.shape[2], x.shape[3]
        x = ops.interpolate(x, size=(h * 2, w * 2))
        x = self.up_0(x, evo)
        h, w = x.shape[2], x.shape[3]
        x = ops.interpolate(x, size=(h * 2, w * 2))
        x = self.up_1(x, evo)
        x = self.up_2(x, evo)
        out = self.conv_img(self.leaky_relu(x))
        return out
        # return x


class GenerationNet(nn.Cell):
    """Generation network"""
    def __init__(self, config):
        super(GenerationNet, self).__init__()
        self.gen_enc = GenerativeEncoder(config.total_length, config.ngf)
        self.gen_dec = GenerativeDecoder(config)
        self.proj = NoiseProjector(config.ngf)

    def construct(self, input_frames, evo_result, noise):
        batch = input_frames.shape[0]
        height = input_frames.shape[2]
        width = input_frames.shape[3]
        evo_feature = self.gen_enc(ops.concat([input_frames, evo_result], axis=1))
        noise_feature = self.proj(noise).reshape(batch, -1, 4, 4, 8, 8)
        noise_feature = noise_feature.permute(0, 1, 4, 5, 2, 3).reshape(batch, -1, height // 8, width // 8)
        feature = ops.concat([evo_feature, noise_feature], axis=1)
        out = self.gen_dec(feature, evo_result)
        return out
