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
"""rbf."""

import math
import mindspore as ms
from mindspore import nn, ops

class CosineCutoff(nn.Cell):
    """Cosine cutoff function."""
    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0):
        super(CosineCutoff, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.cos = ops.Cos()
        self.pi = ms.Tensor(math.pi, dtype=ms.float32)

    def construct(self, distances):
        """Compute the cutoff function."""
        if self.cutoff_lower > 0:
            cutoffs = 0.5 * (
                self.cos(
                    self.pi
                    * (
                        2
                        * (distances - self.cutoff_lower)
                        / (self.cutoff_upper - self.cutoff_lower)
                        + 1.0
                    )
                )
                + 1.0
            )
            # remove contributions below the cutoff radius
            cutoffs = cutoffs * (distances < self.cutoff_upper).astype(ms.float32)
            cutoffs = cutoffs * (distances > self.cutoff_lower).astype(ms.float32)
            return cutoffs

        cutoffs = 0.5 * (self.cos(distances * self.pi / self.cutoff_upper) + 1.0)
        # remove contributions beyond the cutoff radius
        cutoffs = cutoffs * (distances < self.cutoff_upper).astype(ms.float32)
        return cutoffs


class ExpNormalSmearing(nn.Cell):
    """Exponential normal smearing function."""
    def __init__(self, cutoff_lower=0.0, cutoff_upper=10.0, num_rbf=50, trainable=True):
        """Exponential normal smearing function.

        Distances are expanded into exponential radial basis functions.
        Basis function parameters are initialised as proposed by Unke & Mewly 2019 Physnet,
        https://arxiv.org/pdf/1902.08408.pdf .
        A cosine cutoff function is used to ensure smooth transition to 0.

        Args:
            cutoff_lower (float): Lower cutoff radius.
            cutoff_upper (float): Upper cutoff radius.
            num_rbf (int): Number of radial basis functions.
            trainable (bool): Whether the parameters are trainable.
        """
        super(ExpNormalSmearing, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.num_rbf = num_rbf
        self.trainable = trainable

        self.cutoff_fn = CosineCutoff(0, cutoff_upper)
        self.alpha = cutoff_upper / (cutoff_upper - cutoff_lower)

        means, betas = self._initial_params()
        if trainable:
            self.means = ms.Parameter(means, name="means")
            self.betas = ms.Parameter(betas, name="betas")
        else:
            self.means = means
            self.betas = betas

    def _initial_params(self):
        # initialize means and betas according to the default values in PhysNet
        start_value = ms.Tensor(math.exp(-self.cutoff_upper + self.cutoff_lower), dtype=ms.float32)
        means = ms.numpy.linspace(start_value, 1.0, self.num_rbf)
        betas = ms.Tensor([(2 / self.num_rbf * (1 - start_value)) ** -2] * self.num_rbf, dtype=ms.float32)
        return means, betas

    def reset_parameters(self):
        """Reset the parameters to their default values."""
        means, betas = self._initial_params()
        self.means.set_data(means)
        self.betas.set_data(betas)

    def construct(self, dist):
        """Expand incoming distances into basis functions."""
        dist = dist.unsqueeze(-1)
        return self.cutoff_fn(dist) * ms.ops.exp(
            -self.betas
            * (ms.ops.exp(self.alpha * (-dist + self.cutoff_lower)) - self.means) ** 2
        )
