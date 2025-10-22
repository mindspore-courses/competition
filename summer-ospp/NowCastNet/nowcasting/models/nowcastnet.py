import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
import numpy as np
from ..layers.utils import warp, make_grid
from ..layers.generation.generative_network import GenerationNet
from ..layers.evolution.evolution_network import EvolutionNet
from ..layers.generation.noise_projector import NoiseProjector

class Net(nn.Cell):
    """NowcastNet model implementation in MindSpore"""
    def __init__(self, configs):
        super(Net, self).__init__()
        self.configs = configs
        self.pred_length = self.configs.total_length - self.configs.input_length

        # Initialize networks
        self.evo_net = EvolutionNet(self.configs.input_length, self.pred_length, self.configs.img_height, self.configs.img_width)
        self.gen_net = GenerationNet(self.configs)
        self.gen_enc = self.gen_net.gen_enc
        self.gen_dec = self.gen_net.gen_dec
        self.proj = self.gen_net.proj

        self.grid = None
        self.concat = ops.Concat(axis=1)
        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()
        self.expand_dims = ops.ExpandDims()