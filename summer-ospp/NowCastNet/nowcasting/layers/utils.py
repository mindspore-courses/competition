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

def plt_img(field, label, idx, plot_evo=False, evo=None, interval=10, fig_name="", vmin=1, vmax=40, cmap="viridis"):
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    
    def set_enhanced_title(ax, text, color='navy'):
        ax.set_title(text, fontsize=12, fontweight='bold', pad=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.1, edgecolor=color))
        ax.set_axis_off()
    
    alpha = change_alpha(label[idx[0]])
    _ = axs[0][0].imshow(label[idx[0]], alpha=alpha, vmin=vmin, vmax=vmax, cmap=cmap)
    set_enhanced_title(axs[0][0], f"Ground Truth\n{idx[0] * interval + interval} min", 'darkblue')
    
    alpha = change_alpha(label[idx[1]])
    _ = axs[0][1].imshow(label[idx[1]], alpha=alpha, vmin=vmin, vmax=vmax, cmap=cmap)
    set_enhanced_title(axs[0][1], f"Ground Truth\n{idx[1] * interval + interval} min", 'darkblue')
    
    alpha = change_alpha(label[idx[2]])
    _ = axs[0][2].imshow(label[idx[2]], alpha=alpha, vmin=vmin, vmax=vmax, cmap=cmap)
    set_enhanced_title(axs[0][2], f"Ground Truth\n{idx[2] * interval + interval} min", 'darkblue')
    
    alpha = change_alpha(field[idx[0]])
    _ = axs[1][0].imshow(field[idx[0]], alpha=alpha, vmin=vmin, vmax=vmax, cmap=cmap)
    set_enhanced_title(axs[1][0], f"Prediction\n{idx[0] * interval + interval} min", 'darkred')
    
    alpha = change_alpha(field[idx[1]])
    _ = axs[1][1].imshow(field[idx[1]], alpha=alpha, vmin=vmin, vmax=vmax, cmap=cmap)
    set_enhanced_title(axs[1][1], f"Prediction\n{idx[1] * interval + interval} min", 'darkred')
    
    alpha = change_alpha(field[idx[2]])
    _ = axs[1][2].imshow(field[idx[2]], alpha=alpha, vmin=vmin, vmax=vmax, cmap=cmap)
    set_enhanced_title(axs[1][2], f"Prediction\n{idx[2] * interval + interval} min", 'darkred')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, hspace=0.1, wspace=0.05)
    plt.savefig(fig_name, dpi=200, bbox_inches='tight', facecolor='white')
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