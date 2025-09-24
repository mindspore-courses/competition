import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from .module import *
import numpy as np
from mindspore import Parameter

class EvolutionNet(nn.Cell):
    """Evolution network"""
    def __init__(self, t_in, t_out, h_size, w_size, in_channels=32, bilinear=True):
        super(EvolutionNet, self).__init__()
        self.t_in = t_in
        factor = 2 if bilinear else 1
        self.inc = DoubleConv(t_in, in_channels)
        self.down1 = Down(in_channels * 1, in_channels * 2)
        self.down2 = Down(in_channels * 2, in_channels * 4)
        self.down3 = Down(in_channels * 4, in_channels * 8)
        self.down4 = Down(in_channels * 8, in_channels * 16 // factor)

        down_h_size = h_size // 16
        down_w_size = w_size // 16

        self.up1 = Up(in_channels * 16, in_channels * 8 // factor, size=(down_h_size * 2, down_w_size * 2),
                      bilinear=bilinear)
        self.up2 = Up(in_channels * 8, in_channels * 4 // factor, size=(down_h_size * 4, down_w_size * 4),
                      bilinear=bilinear)
        self.up3 = Up(in_channels * 4, in_channels * 2 // factor, size=(down_h_size * 8, down_w_size * 8),
                      bilinear=bilinear)
        self.up4 = Up(in_channels * 2, in_channels, size=(down_h_size * 16, down_w_size * 16), bilinear=bilinear)
        self.outc = OutConv(in_channels, t_out)
        self.gamma = Parameter(Tensor(np.zeros((1, t_out, 1, 1), dtype=np.float32), ms.float32), requires_grad=True)

        self.up1_v = Up(in_channels * 16, in_channels * 8 // factor, size=(down_h_size * 2, down_w_size * 2),
                        bilinear=bilinear)
        self.up2_v = Up(in_channels * 8, in_channels * 4 // factor, size=(down_h_size * 4, down_w_size * 4),
                        bilinear=bilinear)
        self.up3_v = Up(in_channels * 4, in_channels * 2 // factor, size=(down_h_size * 8, down_w_size * 8),
                        bilinear=bilinear)
        self.up4_v = Up(in_channels * 2, in_channels, size=(down_h_size * 16, down_w_size * 16), bilinear=bilinear)
        self.outc_v = OutConv(in_channels, t_out * 2)

    def construct(self, all_frames):
        """evolution construct"""
        x = all_frames[:, :self.t_in]
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        intensity = self.outc(x) * self.gamma

        v = self.up1_v(x5, x4)
        v = self.up2_v(v, x3)
        v = self.up3_v(v, x2)
        v = self.up4_v(v, x1)
        motion = self.outc_v(v)
        return intensity, motion