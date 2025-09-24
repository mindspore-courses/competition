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

    # def construct(self, all_frames):
    #     all_frames = ops.squeeze(all_frames, axis=1)
    #     all_frames = all_frames[:, :, :, :, 0:1]

    #     frames = self.transpose(all_frames, (0, 1, 4, 2, 3))
    #     batch = frames.shape[0]
    #     height = frames.shape[3]
    #     width = frames.shape[4]

    #     # Create grid if not created yet
    #     if self.grid is None:
    #         sample_tensor = Tensor(np.zeros((1, 1, height, width)), dtype=ms.float32)
    #         self.grid = make_grid(sample_tensor)
            
    #     # Input Frames
    #     input_frames = frames[:, :self.configs.input_length]
    #     input_frames = self.reshape(input_frames, (batch, self.configs.input_length, height, width))

    #     # Evolution Network
    #     intensity, motion = self.evo_net(input_frames)
    #     motion_ = self.reshape(motion, (batch, self.pred_length, 2, height, width))
    #     intensity_ = self.reshape(intensity, (batch, self.pred_length, 1, height, width))
        
    #     # Warp frames sequentially
    #     series = []
    #     last_frames = all_frames[:, (self.configs.input_length - 1):self.configs.input_length, :, :, 0]
    #     grid = ops.tile(Tensor(self.grid, ms.float32), (batch, 1, 1, 1))
        
    #     for i in range(self.pred_length):
    #         last_frames = warp(last_frames, motion_[:, i], grid, mode="nearest", padding_mode="border")
    #         last_frames = last_frames + intensity_[:, i]
    #         series.append(last_frames)
            
    #     # Concatenate predicted frames
    #     evo_result = self.concat(series)
        
    #     # Normalize evolution result
    #     evo_result = evo_result / 128.0
        
    #     # Generative Network
    #     evo_feature = self.gen_enc(self.concat([input_frames, evo_result]))
        
    #     # Generate noise and project
    #     noise = Tensor(np.random.randn(batch, self.configs.ngf, height // 32, width // 32), dtype=ms.float32)
    #     noise_feature = self.proj(noise)
        
    #     # Reshape noise feature
    #     noise_feature = self.reshape(noise_feature, (batch, -1, 4, 4, 8, 8))
    #     noise_feature = self.transpose(noise_feature, (0, 1, 4, 5, 2, 3))
    #     noise_feature = self.reshape(noise_feature, (batch, -1, height // 8, width // 8))
        
    #     # Combine features and generate final result
    #     feature = self.concat([evo_feature, noise_feature])
    #     gen_result = self.gen_dec(feature, evo_result)
        
    #     # Add channel dimension
    #     return self.expand_dims(gen_result, -1) 