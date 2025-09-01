# ============================================================================
# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""nn utils"""

from typing import List, Optional, Type
from mindspore import nn, ops, Tensor, context

class SSP(nn.Cell):
    """Shifted Softplus activation function.

    This activation is twice differentiable so can be used when regressing
    gradients for conservative force fields.
    """

    def __init__(self, beta: int = 1, threshold: int = 20):
        super().__init__()
        self.beta = beta
        self.threshold = threshold

    def construct(self, input_x: Tensor) -> Tensor:
        sp0 = ops.softplus(ops.zeros(1), self.beta, self.threshold)
        return ops.softplus(input_x, self.beta, self.threshold) - sp0


def build_mlp(
        input_size: int,
        hidden_layer_sizes: List[int],
        output_size: Optional[int] = None,
        output_activation: Type[nn.Cell] = nn.Identity,
        activation: Type[nn.Cell] = SSP,
        dropout: Optional[float] = None,
) -> nn.Cell:
    """Build a MultiLayer Perceptron.

    Args:
      input_size: Size of input layer.
      hidden_layer_sizes: An array of input size for each hidden layer.
      output_size: Size of the output layer.
      output_activation: Activation function for the output layer.
      activation: Activation function for the hidden layers.
      dropout: Dropout rate for hidden layers.
      checkpoint: Whether to use checkpointing.

    Returns:
      mlp: An MLP sequential container.
    """
    # Size of each layer
    layer_sizes = [input_size] + hidden_layer_sizes
    if output_size:
        layer_sizes.append(output_size)

    # Number of layers
    nlayers = len(layer_sizes) - 1

    # Create a list of activation functions and
    # set the last element to output activation function
    act = [activation for _ in range(nlayers)]
    act[-1] = output_activation

    # Create a list to hold layers
    layers = []
    for i in range(nlayers):
        if dropout is not None:
            layers.append(nn.Dropout(keep_prob=1 - dropout))
        layers.append(nn.Dense(layer_sizes[i], layer_sizes[i + 1]))
        layers.append(act[i]())

    # Create a sequential container
    # if checkpoint:
    #     mlp = CheckpointedSequential(*layers, n_layers=nlayers)
    # else:
    #     mlp = nn.SequentialCell(layers)
    mlp = nn.SequentialCell(layers)
    return mlp


class CheckpointedSequential(nn.Cell):
    """Sequential container with checkpointing."""

    def __init__(self, *args, n_layers: int = 1):
        super().__init__()
        self.n_layers = n_layers
        self.layers = nn.CellList(list(args))

    def construct(self, input_x: Tensor) -> Tensor:
        """Forward pass with checkpointing enabled in training mode."""
        if context.get_context("mode") == context.GRAPH_MODE:
            # In graph mode, checkpointing is handled by MindSpore's graph optimization
            for layer in self.layers:
                input_x = layer(input_x)
        else:
            # In PyNative mode, we can manually checkpoint each layer
            for i in range(self.n_layers):
                input_x = self.layers[i](input_x)
        return input_x
