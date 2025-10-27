# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings
import math
import numpy as np

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor,Parameter
from mindspore.common.initializer import XavierUniform,Constant,initializer

def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n &(n-1) == 0) and n!=0

class MSDeformAttn(nn.Cell):
    def __init__(self,d_model = 256, n_levels = 4,n_heads=8,n_points=4):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super(MSDeformAttn, self).__init__()

        if d_model % n_heads != 0:
            raise ValueError(
                'd_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads)
            )

        _d_per_head = d_model // n_heads

        # Better set _d_per_head to a power of 2 which is more efficient in our implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn(
                "You'd better set d_model in MSDeformAttn to make the dimension"
                "of each attention head a power of 2 which is more efficient in"
                "our implementation."
            )

            self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Dense(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Dense(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Dense(d_model, d_model)
        self.output_proj = nn.Dense(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        # Initialize sampling_offsets.weight with zeros
        self.sampling_offsets.weight.set_data(initializer(Constant(0.), self.sampling_offsets.weight.shape))
        # Initialize sampling_offsets.bias with grid_init
        thetas = np.arange(self.n_heads, dtype=np.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = np.stack([np.cos(thetas),np.sin(thetas)],-1)
        grid_init =  (grid_init/np.max(np.abs(grid_init),axis=-1,keepdims=True))\
        .reshape(self.n_heads,1,1,2).repeat(self.n_levels,axis=1).repeat(self.n_points,axis=2)

        for i in range(self.n_points):
            grid_init[:,:,i,:] *= i+1
        
        grid_init = grid_init.reshape(-1)
        self.sampling_offsets.bias.set_data(Tensor(grid_init,ms.float32))

        # Initialize attention_weights.weight and bias with zeros
        self.attention_weights.weight.set_data(initializer(Constant(0.),
            self.attention_weights.weight.shape))
        self.attention_weights.bias.set_data(initializer(Constant(0.), self.attention_weights.bias.shape))

        # Initialize value_proj.weight with XavierUniform and bias with zeros
        self.value_proj.weight.set_data(initializer(XavierUniform(), self.value_proj.weight.shape))
        self.value_proj.bias.set_data(initializer(Constant(0.), self.value_proj.bias.shape))

        # Initialize output_proj.weight with XavierUniform and bias with zeros
        self.output_proj.weight.set_data(initializer(XavierUniform(), self.output_proj.weight.shape))
        self.output_proj.bias.set_data(initializer(Constant(0.), self.output_proj.bias.shape))

    def construct(self,
                  query,
                  reference_points,
                  input_flatten,
                  input_spatial_shapes,
                  input_level_start_index,
                  input_padding_mask=None
                  ):
        
        """
        Arguments:
            query: (N, Length_{query}, C)
            reference_points: (N, Length_{query}, n_levels, 2),
                range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
            input_flatten: (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
            input_spatial_shapes: (n_levels,2),
                [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            input_level_start_index: (n_levels, ),
                [0, H_0*W_0, H_0*W_0+H_1*W_1, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
            input_padding_mask: (N, \sum_{l=0}^{L-1} H_l \cdot W_l),
                True for padding elements, False for non-padding elements

        Returns:
            output: (N,Length_{query},C)
        """
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        #Check input data
        assert ops.reduce_sum(input_spatial_shapes[:,0]*input_spatial_shapes[:,1]) == Len_in
        assert reference_points.shape[2] == \
               input_spatial_shapes.shape[0] == \
               input_level_start_index.shape[0] == \
               self.n_levels
        
        value = self.value_proj(input_flatten)

        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query) \
            .view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query) \
            .view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = ops.softmax(attention_weights, -1) \
            .view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        
        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = ops.Stack(-1)(
                [input_spatial_shapes[...,-1],input_spatial_shapes[...,0]]
            )
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                + sampling_offsets \
                                / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1]==4:
            sampling_locations = reference_points[:, :, None, :, :2]+ sampling_offsets\
                                /self.n_points*reference_points[:, :, None, :, None,2:] * 0.5
        
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {}" \
                "instead.".format(reference_points.shape[-1])
            )

        # Note: The following part requires a custom operator (MSDeformAttnFunction) in MindSpore.
        # You need to implement it separately using MindSpore's Custom operator or Primitive.
        # For simplicity, we assume it's implemented as a function here.
        output = ms_deform_attn_core_pytorch(
            value,
            input_spatial_shapes,
            sampling_locations,
            attention_weights
        )

        output = self.output_proj(output)

        return output
    

def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    # for debug and test only,
    # need to use cuda version instead
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    # 把value分割到各个特征层上得到对应的 list value
    value_list = value.split([int(H_ * W_) for H_, W_ in value_spatial_shapes], dim=1)
    # 采样点坐标从[0,1] -> [-1, 1]  F.grid_sample要求采样坐标归一化到[-1, 1]
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = ops.flatten(value_list[lid_],start_dim=2).transpose(1, 2).reshape(int(N_*M_), int(D_), int(H_), int(W_))  # 得到每个特征层的value list
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)  # 得到每个特征层的采样点 list
        # N_*M_, D_, Lq_, P_  采样算法  根据每个特征层采样点到每个特征层的value进行采样  非采样点用0填充
        sampling_value_l_ = ops.grid_sample(value_l_, sampling_grid_l_,
                                          mode='bilinear', padding_mode='zeros', align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.transpose(1, 2).reshape(N_*M_, 1, Lq_, L_*P_)
    # 注意力权重 和 采样后的value 进行 weighted sum
    output = (ops.stack(sampling_value_list, axis=-2).flatten(-2) * attention_weights).sum(-1).view(N_, M_*D_, Lq_)
    return output.transpose(1, 2).contiguous()