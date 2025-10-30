import mindspore as ms
from mindspore import nn

def print_mindspore_model_params(network: nn.Cell):
    print("===== MindSpore 模型参数 =====")
    for param in network.get_parameters():
        print(f"{param.name}: {tuple(param.shape)}")

from models import *
model = MVSNet(refine=False)
print_mindspore_model_params(model)