from typing import Any, Callable, Dict, List, Optional, OrderedDict
import mindspore as ms
from mindspore import nn, ops
from mindspore.common.initializer import initializer,XavierUniform


class Conv2dNormActivation(nn.Cell):
    """Conv2d with normalization and activation"""
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 
                             kernel_size=kernel_size, 
                             stride=stride, 
                             pad_mode='pad',
                             padding=padding,
                             weight_init=XavierUniform(),
                             has_bias=True)
        # self.norm = nn.BatchNorm2d(out_channels)
        # self.act = nn.ReLU()

    def construct(self, x):
        x = self.conv(x)
        # x = self.norm(x)
        #x = self.act(x)
        return x

class FeaturePyramidNetwork(nn.Cell):
    """FPN implementation in MindSpore"""
    def __init__(self, in_channels_list=[3, 256, 512, 1024, 2048], out_channels=16):
        super().__init__()
        self.inner_blocks = nn.CellList()
        self.layer_blocks = nn.CellList()
        
        # Inner blocks (1x1 conv for channel reduction)
        for in_channels in in_channels_list:
            self.inner_blocks.append(
                Conv2dNormActivation(in_channels, out_channels, kernel_size=1,padding=0))
            self.layer_blocks.append(
                Conv2dNormActivation(out_channels, out_channels, kernel_size=3, padding=1))


    def get_result_from_inner_blocks(self, x: ms.Tensor, idx: int) -> ms.Tensor:
        """
        This is equivalent to self.inner_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.inner_blocks)
        if idx < 0:
            idx += num_blocks
        out = x
        for i, module in enumerate(self.inner_blocks):
            if i == idx:
                out = module(x)
        return out

    def get_result_from_layer_blocks(self, x: ms.Tensor, idx: int) -> ms.Tensor:
        """
        This is equivalent to self.layer_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.layer_blocks)
        if idx < 0:
            idx += num_blocks
        out = x
        for i, module in enumerate(self.layer_blocks):
            if i == idx:
                out = module(x)
        return out

    def construct(self, x):
        """
        Computes the FPN for a set of feature maps.

        Args:
            x (OrderedDict[Tensor]): feature maps for each feature level.

        Returns:
            results (OrderedDict[Tensor]): feature maps after FPN layers.
                They are ordered from the highest resolution first.
        """
        # unpack OrderedDict into two lists for easier handling
        names = list(x.keys())
        x = list(x.values())

        last_inner = self.get_result_from_inner_blocks(x[-1], -1)
        results = []
        results.append(self.get_result_from_layer_blocks(last_inner, -1))

        for idx in range(len(x) - 2, -1, -1):
            inner_lateral = self.get_result_from_inner_blocks(x[idx], idx)
            feat_shape = inner_lateral.shape[-2:]
            inner_top_down = ops.interpolate(last_inner, size=feat_shape, mode="nearest")
            last_inner = inner_lateral + inner_top_down
            results.insert(0, self.get_result_from_layer_blocks(last_inner, idx))
         
        # make it back an OrderedDict
        out = OrderedDict([(k, v) for k, v in zip(names, results)])

        return out

class FPN(nn.Cell):
    """Wrapper for FPN with input preprocessing"""
    def __init__(self,
                 config,
                 channel_last=True):
        super().__init__()
        self.fpn = FeaturePyramidNetwork(config['in_channels_list'],config['out_channels'])
        self.channel_last = channel_last

    @staticmethod
    def _to_channel_first(batch: Dict[str, ms.Tensor]) -> Dict[str, ms.Tensor]:
        return OrderedDict({k: v.movedim(-1, 1) for k, v in batch.items()})

    @staticmethod
    def _to_channel_last(batch: Dict[str, ms.Tensor]) -> Dict[str, ms.Tensor]:
        return OrderedDict({k: v.movedim(1, -1) for k, v in batch.items()})
        
    def construct(self, 
                  batch: Dict[str, ms.Tensor]) -> Dict[str, ms.Tensor]:
        # Assume x is a list of feature maps from backbone
        
        # Adjust channel format
        if self.channel_last:
            batch = self._to_channel_first(batch)

        # Align features
        batch = self.fpn(batch)

        # Adjust channel format
        if self.channel_last:
            batch = self._to_channel_last(batch)

        return batch

    @classmethod
    def from_config(cls,config:Dict[str,Any])->"FPN":
        """Initialize FPN from a config dictionary."""
        return cls(config)
    
def build_fpn(name: str, *args, **kwargs) -> FPN:
    if 'fpn' in name.lower():
        return FPN.from_config(*args, **kwargs)
    raise ValueError(f"Unsupported module: {name}")