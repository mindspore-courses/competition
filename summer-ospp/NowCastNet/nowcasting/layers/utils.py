import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as mnp
from mindspore import Tensor, Parameter
from mindspore.common.initializer import initializer, Normal
import numpy as np
import matplotlib.pyplot as plt

def make_grid(inputs):
    """get 2D grid"""
    batch_size, _, height, width = inputs.shape
    xx = np.arange(0, width).reshape(1, -1)
    xx = np.tile(xx, (height, 1))
    yy = np.arange(0, height).reshape(-1, 1)
    yy = np.tile(yy, (1, width))
    xx = xx.reshape(1, 1, height, width)
    xx = np.tile(xx, (batch_size, 1, 1, 1))
    yy = yy.reshape(1, 1, height, width)
    yy = np.tile(yy, (batch_size, 1, 1, 1))
    grid = np.concatenate((xx, yy), axis=1).astype(np.float32)
    return grid

def warp(inputs, flow, grid, mode="bilinear", padding_mode="zeros"):
    width = inputs.shape[-1]
    vgrid = grid + flow
    vgrid = 2.0 * vgrid / max(width - 1, 1) - 1.0
    vgrid = vgrid.transpose(0, 2, 3, 1)
    output = ops.grid_sample(inputs, vgrid, padding_mode=padding_mode, mode=mode, align_corners=True)
    return output

def change_alpha(x):
    alpha = np.zeros(x.shape)
    alpha[x >= 2] = 1
    return alpha

def plt_img(field, label, interval=10, fig_name="", vmin=1, vmax=40, cmap="viridis"):
    target_idx = 0
    _, axs = plt.subplots(1, 2, figsize=(10, 4))
    
    alpha = change_alpha(label[target_idx])
    _ = axs[0].imshow(label[target_idx], alpha=alpha, vmin=vmin, vmax=vmax, cmap=cmap)
    axs[0].set_title(f"Ground Truth - {interval} min")
    axs[0].set_axis_off()
    
    alpha = change_alpha(field[target_idx])
    _ = axs[1].imshow(field[target_idx], alpha=alpha, vmin=vmin, vmax=vmax, cmap=cmap)
    axs[1].set_title(f"Prediction - {interval} min")
    axs[1].set_axis_off()
    
    plt.tight_layout()
    plt.savefig(fig_name, dpi=180, bbox_inches='tight')
    plt.show()
    plt.close()

class SpectralNormal(nn.Cell):
    def __init__(self, module, n_power_iterations=1, dim=0, eps=1e-12, n1=1.0, n2=0):
        super(SpectralNormal, self).__init__()
        self.parametrizations = module
        self.weight = module.weight
        ndim = self.weight.ndim
        if dim >= ndim or dim < -ndim:
            raise IndexError("Dimension out of range (expected to be in range of "
                             f"[-{ndim}, {ndim - 1}] but got {dim})")

        if n_power_iterations <= 0:
            raise ValueError('Expected n_power_iterations to be positive, but '
                             'got n_power_iterations={}'.format(n_power_iterations))
        self.dim = dim if dim >= 0 else dim + ndim
        self.eps = eps
        self.l2_normalize = ops.L2Normalize(epsilon=self.eps)
        self.expand_dims = ops.ExpandDims()
        self.assign = ops.Assign()
        if ndim > 1:
            self.n_power_iterations = n_power_iterations
            weight_mat = self._reshape_weight_to_matrix()
            h, w = weight_mat.shape
            u = initializer(Normal(n1, n2), [h]).init_data()
            v = initializer(Normal(n1, n2), [w]).init_data()
            self._u = Parameter(self.l2_normalize(u), requires_grad=False)  # 封装成Parameter对象
            self._v = Parameter(self.l2_normalize(v), requires_grad=False)

    def construct(self, *inputs, **kwargs):
        if self.weight.ndim == 1:
            self.l2_normalize(self.weight)
            self.assign(self.parametrizations.weight, self.weight)
        else:
            weight_mat = self._reshape_weight_to_matrix()
            if self.training:
                self._u, self._v = self._power_method(weight_mat, self.n_power_iterations)
            u = self._u.copy()
            v = self._v.copy()
            sigma = ops.tensor_dot(u, mnp.multi_dot([weight_mat, self.expand_dims(v, -1)]), 1)
            self.assign(self.parametrizations.weight, self.weight / sigma)
        return self.parametrizations(*inputs, **kwargs)

    def _power_method(self, weight_mat, n_power_iterations):
        for _ in range(n_power_iterations):
            self._u = self.l2_normalize(mnp.multi_dot([weight_mat, self.expand_dims(self._v, -1)]).flatten())
            self._u += 0
            self._v = self.l2_normalize(mnp.multi_dot([weight_mat.T, self.expand_dims(self._u, -1)]).flatten())
        return self._u, self._v

    def _reshape_weight_to_matrix(self):
        if self.dim != 0:
            input_perm = [d for d in range(self.weight.dim()) if d != self.dim]
            input_perm.insert(0, self.dim)
            self.weight = ops.transpose(self.weight, input_perm)
        return self.weight.reshape(self.weight.shape[0], -1)

