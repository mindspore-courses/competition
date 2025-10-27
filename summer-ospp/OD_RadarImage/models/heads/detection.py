from __future__ import annotations  # noqa: F407

from collections import OrderedDict
from typing import Any, Dict, Optional

import mindspore as ms

from mindspore import nn

from models.layers.unary import Unary1d

class UnaryDetectionHead(nn.Cell):
    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 num_reg_layers: int = 1,
                 num_cls_layers: int = 1,
                 bias: Optional[bool] = False,
                 dropout: float = 0.0,
                 channels_last: Optional[bool] = True,
                 **kwargs) -> None:
        # Initialize parent class
        super().__init__()

        # Initialize instance attributes
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_reg_layers = num_reg_layers
        self.num_cls_layers = num_cls_layers
        self.bias = bias
        self.dropout = dropout
        self.channels_last = channels_last

        # Define actiavtion functions
        self.activations = {
            'center': 'Identity',
            'size': 'ReLU',
            'angle': 'Tanh',
            'class': 'Identity'
        }

        # Initialize instance layers
        self.layers = nn.CellDict({
            'center_head':self._get_reg_branch(3),
            'size_head': self._get_reg_branch(3),
            'angle_head': self._get_reg_branch(2),
            'class_head': self._get_cls_branch(self.num_classes)
        })

        # Initialize activation functions
        self.activation_fn = nn.CellDict(
            {k:self._get_activation_fn(v) for k,v in self.activations.items()}
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> UnaryDetectionHead:  # noqa: F821
        return cls(
            config['in_channels'],
            config['num_classes'],
            config.get('num_reg_layers', 1),
            config.get('num_cls_layers', 1),
            config.get('bias', False),
            config.get('dropout', 0.0),
            config.get('channels_last', True)
        )
    
    def _get_activation_fn(self, name: str) -> nn.Cell:
        if 'softmax' in name.lower() and self.channels_last:
            return getattr(nn, name)(dim=-1)
        if 'softmax' in name.lower():
            return getattr(nn, name)(dim=1)
        return getattr(nn, name)()
    
    def _get_cls_branch(self, out_channels:int) -> nn.Cell:
        """Returns a sequence of unary layers

        Arguments:
            out_channels: Number of output channles for the last layer
                of the sequence.

        Returns:
            A sequence of unary layers.
        """
        cls_branch = []
        for _ in range(self.num_reg_layers - 1):
            cls_branch.append(Unary1d(self.in_channels,self.in_channels,
                                      bias=self.bias,channels_last=self.channels_last))
            cls_branch.append(nn.ReLU())
            cls_branch.append(nn.Dropout(self.dropout))

        cls_branch.append(Unary1d(self.in_channels,out_channels,bias=self.bias,channels_last=self.channels_last))

        return nn.SequentialCell(*cls_branch)
    
    def _get_reg_branch(self, out_channels: int) -> nn.Cell:
        """Returns a sequence of unary layers with actiavtion function

        Arguments:
            out_channels: Number of output channels for the last layer
                of the sequence.

        Returns:
            A sequence of unary layers with activation function, wheras the last layer does not
            have an actiavtion function.
        """

        for _ in range(self.num_reg_layers - 1):
            reg_branch =[]
            reg_branch.append(Unary1d(self.in_channels, self.in_channels,
                                      bias=self.bias, channels_last=self.channels_last))
            reg_branch.append(nn.ReLU())
            reg_branch.append(nn.Dropout(self.dropout))

        reg_branch.append(Unary1d(self.in_channels, out_channels,
                                  bias=self.bias, channels_last=self.channels_last))

        return nn.Sequential(*reg_branch)
    
    def construct(self, batch: ms.Tensor,
                ref: Dict[str, ms.Tensor])-> Dict[str,ms.Tensor]:
        """Returns bounding box predictions given a feature tensor and reference points.
        
        Arguments:
            batch: Batched feature tensor with shape (B, N, in_channels)
            ref: Ordered dictionary that contains at least this entry.
            "center": Bounding box center coordinates of shape (B, N, 3).

        Returns:
            out: Ordered dictonary that contains these entries:
                "class": Bounding box class probbalites of shape (B, N, num_class)
                "center": Bounding box center coordinates of shape (B, N, 3).
                "size": Bounding box size values of shape (B, N, 3).
                "angle": Bounding box orientation values of shape (B, N, 2).

        """

        # Apply layers and activations
        iterator = zip(self.activation_fn.items(),self.layers.values())

        out = OrderedDict({
            k:activation(layer(batch)) for (k, activation), layer in iterator
        })

        # Add reference position to relative center position
        out['center'][..., :3] += ref['center'][..., :3]

        return out

class LinearDetectionHead(nn.Cell):
    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 num_reg_layers: int = 1,
                 num_cls_layers: int = 1,
                 bias: Optional[bool] = False,
                 dropout: float = 0.0,
                 channels_last: Optional[bool] = True,
                 **kwargs) -> None:
        # Initialize parent class
        super().__init__()

        # Initialize instance attributes
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_reg_layers = num_reg_layers
        self.num_cls_layers = num_cls_layers
        self.bias = bias
        self.dropout = dropout
        self.channels_last = channels_last

        # Define activation functions
        self.activations = {
            'center': 'Identity',
            'size': 'ReLU',
            'angle': 'Tanh',
            'class': 'Identity'
        }

        # Initialize instance layers
        self.layers = nn.CellDict({
            'center_head': RegressionBranch(in_channels, 3, num_reg_layers),
            'size_head': RegressionBranch(in_channels, 3, num_reg_layers),
            'angle_head': RegressionBranch(in_channels, 2, num_reg_layers),
            'class_head': ClassificationBranch(in_channels, num_classes, num_cls_layers),
        })


        # Initialize activation functions
        self.activation_fn = nn.CellDict(
            {k: self._get_activation_fn(v) for k, v in self.activations.items()}
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> LinearDetectionHead:  # noqa: F821
        return cls(
            config['in_channels'],
            config['num_classes'],
            config.get('num_reg_layers', 1),
            config.get('num_cls_layers', 1),
            config.get('bias', False),
            config.get('dropout', 0.0),
            config.get('channels_last', True)
        )

    def _get_activation_fn(self, name: str) -> nn.Cell:
        if 'softmax' in name.lower() and self.channels_last:
            return getattr(nn, name)(dim=-1)
        if 'softmax' in name.lower():
            return getattr(nn, name)(dim=1)
        return getattr(nn, name)()

    def construct(self, batch: ms.Tensor, ref: ms.Tensor) -> Dict[str, ms.Tensor]:
        """
        Arguments:
            batch: Batched feature tensor with shape (B, N, in_channels).
            ref: Reference bounding box positions with shape (B, N, 3).

        Returns:
            out: Ordered dictionary that contains these entries:
                "class": Bounding box class probabilities of shape (B, N, num_classes)
                "center": Bounding box center coordinates of shape (B, N, 3).
                "size": Bounding box size values of shape (B, N, 3).
                "angle": Bounding box orientation values of shape (B, N, 2).
        """
        # Apply layers and activations
        iterator = zip(self.activation_fn.items(), self.layers.values())

        out = OrderedDict(
            {k: activation(layer(batch)) for (k, activation), layer in iterator}
        )

        # Add reference position to relative center position
        out['center'][..., :3] += ref['center'][..., :3]

        return out



class RegressionBranch(nn.Cell):
    def __init__(self, in_channels, out_channels, num_layers):
        super().__init__()
        self.layers = nn.CellList()
        for _ in range(num_layers - 1):
            self.layers.append(nn.Dense(in_channels, in_channels,has_bias=False))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Dense(in_channels, out_channels,has_bias=False))

    def construct(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class ClassificationBranch(nn.Cell):
    def __init__(self, in_channels, out_channels, num_layers):
        super().__init__()
        self.layers = nn.CellList()
        for _ in range(num_layers - 1):
            self.layers.append(nn.Dense(in_channels, in_channels,has_bias=False))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Dense(in_channels, out_channels,has_bias=False))

    def construct(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    
def build_detection_head(name: str, *args, **kwargs) -> nn.Cell:
    if 'unary' in name.lower():
        return UnaryDetectionHead.from_config(*args, **kwargs)

    if 'linear' in name.lower():
        return LinearDetectionHead.from_config(*args, **kwargs)
