import os
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
import numpy as np
from nowcasting.models import nowcastnet

class Model(object):
    """Model factory for NowcastNet"""
    def __init__(self, configs):
        self.configs = configs
        networks_map = {
            'NowcastNet': nowcastnet.Net,
        }
        self.data_frame = []
        
        if configs.model_name in networks_map:
            Network = networks_map[configs.model_name]
            self.network = Network(configs)
            self.test_load()
        else:
            raise ValueError('Name of network unknown %s' % configs.model_name)

    def test_load(self):
        """Load pretrained weights from converted checkpoint"""
        evo_params = load_checkpoint(self.configs.evo_pretrained_model)
        evo_model = self.network.evo_net
        load_param_into_net(evo_model, evo_params)

        gen_params = load_checkpoint(self.configs.gen_pretrained_model)
        gen_model = self.network.gen_net
        load_param_into_net(gen_model, gen_params)

    def test(self, frames):
        """Run inference on input frames"""
        frames_tensor = Tensor(frames, dtype=ms.float32)
        self.network.set_train(False)
        next_frames = self.network(frames_tensor)
        return next_frames.asnumpy() 