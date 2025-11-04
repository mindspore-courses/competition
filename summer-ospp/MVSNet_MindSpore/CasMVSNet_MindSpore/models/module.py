import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as mnp
import numpy as np
import cv2  # 用于 grid_sample 替代
from mindspore.common.initializer import One,Zero, HeUniform, XavierUniform, Uniform,initializer

import sys
sys.path.append("..")
from utils import local_pcd

def init_bn(module):
    if module.gamma is not None:
        module.gamma.set_data(initializer(One(), module.gamma.shape, module.gamma.dtype))
    if module.beta is not None:
        module.beta.set_data(initializer(Zero(), module.beta.shape, module.beta.dtype))
    return

def init_uniform(module, init_method):
    if module.weight is not None:
        if init_method == "kaiming":
            module.weight.set_data(initializer(HeUniform(),module.weight.shape,module.weight.dtype))
        elif init_method == "xavier":
            module.weight.set_data(initializer(XavierUniform(),module.weight.shape,module.weight.dtype))
    return module


class ConvBnReLU(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, pad_mode='pad', has_bias=False)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.1, eps=1e-5)
    def construct(self, x):
        return ops.relu(self.bn(self.conv(x)))


class ConvBn(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBn, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, pad_mode='pad', has_bias=False)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.1, eps=1e-5)

    def construct(self, x):
        return self.bn(self.conv(x))


class ConvBnReLU3D(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, pad_mode='pad', has_bias=False)
        self.bn = nn.BatchNorm3d(out_channels, momentum=0.1, eps=1e-5)
        
    def construct(self, x):
        return ops.relu(self.bn(self.conv(x)))


class ConvBn3D(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBn3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, pad_mode='pad', has_bias=False)
        self.bn = nn.BatchNorm3d(out_channels, momentum=0.1, eps=1e-5)
        
    def construct(self, x):
        return self.bn(self.conv(x))


class BasicBlock(nn.Cell):
    def __init__(self, in_channels, out_channels, stride, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = ConvBnReLU(in_channels, out_channels, kernel_size=3, stride=stride, pad=1)
        self.conv2 = ConvBn(out_channels, out_channels, kernel_size=3, stride=1, pad=1)
        self.downsample = downsample
        self.stride = stride

    def construct(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        return out
class Hourglass3d(nn.Cell):
    def __init__(self, channels):
        super(Hourglass3d, self).__init__()
        self.conv1a = ConvBnReLU3D(channels, channels * 2, kernel_size=3, stride=2, pad=1)
        self.conv1b = ConvBnReLU3D(channels * 2, channels * 2, kernel_size=3, stride=1, pad=1)
        self.conv2a = ConvBnReLU3D(channels * 2, channels * 4, kernel_size=3, stride=2, pad=1)
        self.conv2b = ConvBnReLU3D(channels * 4, channels * 4, kernel_size=3, stride=1, pad=1)
        self.dconv2 = nn.SequentialCell(
            nn.Conv3dTranspose(channels * 4, channels * 2, kernel_size=3, padding=1, output_padding=1, stride=2,
                               has_bias=False, dtype=ms.float16),
            nn.BatchNorm3d(channels * 2))
        self.dconv1 = nn.SequentialCell(
            nn.Conv3dTranspose(channels * 2, channels, kernel_size=3, padding=1, output_padding=1, stride=2,
                               has_bias=False, dtype=ms.float16),
            nn.BatchNorm3d(channels))
        self.redir1 = ConvBn3D(channels, channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = ConvBn3D(channels * 2, channels * 2, kernel_size=1, stride=1, pad=0)
        
    def construct(self, x):
        conv1 = self.conv1b(self.conv1a(x))
        conv2 = self.conv2b(self.conv2a(conv1))
        dconv2 = ops.relu(self.dconv2(conv2) + self.redir2(conv1))
        dconv1 = ops.relu(self.dconv1(dconv2) + self.redir1(x))
        return dconv1

def homo_warping(src_fea, src_proj, ref_proj, depth_values):
    """
    MindSpore implementation of homo_warping (Torch equivalent)
    Args:
        src_fea: [B, C, H, W]
        src_proj: [B, 4, 4]
        ref_proj: [B, 4, 4]
        depth_values: [B, Ndepth]
    Returns:
        warped_src_fea: [B, C, Ndepth, H, W]
    """
    # === disable gradients (equiv. to torch.no_grad) ===
    src_proj = ops.stop_gradient(src_proj)
    ref_proj = ops.stop_gradient(ref_proj)
    depth_values = ops.stop_gradient(depth_values)
    batch, channels = src_fea.shape[0], src_fea.shape[1]
    num_depth = depth_values.shape[1]
    height, width = src_fea.shape[2], src_fea.shape[3]
    # === projection matrices ===
    proj = ops.matmul(src_proj, ops.inverse(ref_proj))
    rot = proj[:, :3, :3]   # [B,3,3]
    trans = proj[:, :3, 3:4]  # [B,3,1]
    # === image grid ===
    y, x = ops.meshgrid(mnp.arange(0, height, dtype=ms.float32),
                        mnp.arange(0, width, dtype=ms.float32),
                        indexing='ij')
    y, x = y.reshape(-1), x.reshape(-1)
    xyz = ops.stack((x, y, ops.ones_like(x)), 0)  # [3, H*W]
    xyz = ops.expand_dims(xyz, 0).tile((batch, 1, 1))  # [B, 3, H*W]
    # === rotation ===
    rot_xyz = ops.matmul(rot, xyz)  # [B, 3, H*W]
    rot_depth_xyz = ops.expand_dims(rot_xyz, 2).tile((1, 1, num_depth, 1))  # [B, 3, Ndepth, H*W]
    # === apply depth ===
    depth_values = ops.expand_dims(depth_values, -1)  # [B, Ndepth, 1]
    depth_values = ops.expand_dims(depth_values, 1)   # [B, 1, Ndepth, 1]
    rot_depth_xyz = rot_depth_xyz * depth_values      # [B, 3, Ndepth, H*W]
    # === project to reference view ===
    proj_xyz = rot_depth_xyz + ops.expand_dims(trans, 2)  # [B, 3, Ndepth, H*W]
    denom = proj_xyz[:, 2:3, :, :]
    denom = ops.select(ops.equal(denom, 0.0), denom + 1e-8, denom)
    proj_xy = proj_xyz[:, :2, :, :] / denom  # [B, 2, Ndepth, H*W]
    # === normalize to [-1, 1] ===
    proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
    proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
    proj_xy = ops.stack((proj_x_normalized, proj_y_normalized), axis=-1)  # [B, Ndepth, H*W, 2]
    # === reshape for grid_sample ===
    grid = proj_xy.view(batch, num_depth * height, width, 2)
    # === bilinear sampling ===
    warped_src_fea = ops.grid_sample(src_fea,
                                     grid,
                                     mode='bilinear',
                                     padding_mode='zeros',
                                     align_corners=False)  # match torch default
    warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)
    return warped_src_fea



class Conv2d(nn.Cell):
    """Applies a 2D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Cell): convolution module
        bn (nn.Cell): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, norm_type='BN', init_method="xavier", **kwargs):
        super(Conv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,has_bias=(not bn), pad_mode='pad', **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.relu = nn.ReLU()
        if norm_type == 'IN':
            self.bn = nn.InstanceNorm2d(out_channels, momentum=bn_momentum) if bn else None
        elif norm_type == 'BN':
            self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.ifrelu = relu
    def construct(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.ifrelu:
            x = self.relu(x)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)



class Deconv2d(nn.Cell):
    """Applies a 2D deconvolution (optionally with batch normalization and relu activation)
       over an input signal composed of several input planes.

       Attributes:
           conv (nn.Cell): convolution module
           bn (nn.Cell): batch normalization module
           relu (bool): whether to activate by relu

       Notes:
           Default momentum for batch normalization is set to be 0.01,

       """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Deconv2d, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2]
        self.stride = stride
        self.conv = nn.Conv2dTranspose(in_channels, out_channels, kernel_size, stride=stride,
                                       has_bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.ifrelu = relu
        self.relu = nn.ReLU()

    def construct(self, x):
        y = self.conv(x)
        if self.stride == 2:
            h, w = list(x.size())[2:]
            y = y[:, :, :2 * h, :2 * w].contiguous()
        if self.bn is not None:
            x = self.bn(y)
        if self.ifrelu:
            x = self.relu(x)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)

class Conv3d(nn.Cell):
    """Applies a 3D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Cell): convolution module
        bn (nn.Cell): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Conv3d, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride,
                              has_bias=(not bn), pad_mode='pad', **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum) if bn else None
        self.ifrelu = relu
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.ifrelu:
            x = self.relu(x)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)

class Deconv3d(nn.Cell):
    """Applies a 3D deconvolution (optionally with batch normalization and relu activation)
       over an input signal composed of several input planes.

       Attributes:
           conv (nn.Cell): convolution module
           bn (nn.Cell): batch normalization module
           relu (bool): whether to activate by relu

       Notes:
           Default momentum for batch normalization is set to be 0.01,

       """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Deconv3d, self).__init__()
        self.out_channels = out_channels
        self.stride = stride
        self.conv = nn.Conv3dTranspose(in_channels, out_channels, kernel_size, stride=stride,
                                       has_bias=(not bn), pad_mode='pad', **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum) if bn else None
        self.ifrelu = relu
        self.relu = nn.ReLU()
    def construct(self, x):
        y = self.conv(x)
        if self.bn is not None:
            x = self.bn(y)
        if self.ifrelu:
            x = self.relu(x)
        return x
    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)


class DeConv2dFuse(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size, relu=True, bn=True, bn_momentum=0.1):
        super().__init__()
        self.deconv = Deconv2d(in_channels, out_channels, kernel_size, stride=2, padding=1, output_padding=1,
                               bn=True, relu=relu, bn_momentum=bn_momentum)
        self.conv = Conv2d(2 * out_channels, out_channels, kernel_size, stride=1, padding=1,
                           bn=bn, relu=relu, bn_momentum=bn_momentum)
        self.concat = ops.Concat(axis=1)

    def construct(self, x_pre, x):
        x = self.deconv(x)
        x = self.concat((x, x_pre))
        x = self.conv(x)
        return x

class FeatureNet(nn.Cell):
    def __init__(self, base_channels, num_stage=3, stride=4, arch_mode="unet"):
        super().__init__()
        assert arch_mode in ["unet", "fpn"], f"mode must be in 'unet' or 'fpn', but get:{arch_mode}"
        print(f"*************feature extraction arch mode:{arch_mode}****************")
        self.arch_mode = arch_mode
        self.stride = stride
        self.base_channels = base_channels
        self.num_stage = num_stage

        self.conv0 = nn.SequentialCell(
            Conv2d(3, base_channels, 3, 1, padding=1),
            Conv2d(base_channels, base_channels, 3, 1, padding=1),
        )

        self.conv1 = nn.SequentialCell(
            Conv2d(base_channels, base_channels * 2, 5, stride=2, padding=2),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
        )

        self.conv2 = nn.SequentialCell(
            Conv2d(base_channels * 2, base_channels * 4, 5, stride=2, padding=2),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
        )

        self.out1 = nn.Conv2d(base_channels * 4, base_channels * 4, 1, has_bias=False)
        self.out_channels = [4 * base_channels]

        if self.arch_mode == 'unet':
            if num_stage == 3:
                self.deconv1 = DeConv2dFuse(base_channels * 4, base_channels * 2, 3)
                self.deconv2 = DeConv2dFuse(base_channels * 2, base_channels, 3)

                self.out2 = nn.Conv2d(base_channels * 2, base_channels * 2, 1, has_bias=False)
                self.out3 = nn.Conv2d(base_channels, base_channels, 1, has_bias=False)
                self.out_channels += [2 * base_channels, base_channels]

            elif num_stage == 2:
                self.deconv1 = DeConv2dFuse(base_channels * 4, base_channels * 2, 3)
                self.out2 = nn.Conv2d(base_channels * 2, base_channels * 2, 1, has_bias=False)
                self.out_channels += [2 * base_channels]

        elif self.arch_mode == "fpn":
            final_chs = base_channels * 4
            if num_stage == 3:
                self.inner1 = nn.Conv2d(base_channels * 2, final_chs, 1, has_bias=True)
                self.inner2 = nn.Conv2d(base_channels * 1, final_chs, 1, has_bias=True)

                self.out2 = nn.Conv2d(final_chs, base_channels * 2, 3, padding=1, pad_mode='pad', has_bias=False)
                self.out3 = nn.Conv2d(final_chs, base_channels, 3, padding=1, pad_mode='pad', has_bias=False)
                self.out_channels += [base_channels * 2, base_channels]

            elif num_stage == 2:
                self.inner1 = nn.Conv2d(base_channels * 2, final_chs, 1, has_bias=True)
                self.out2 = nn.Conv2d(final_chs, base_channels, 3, padding=1, pad_mode='pad', has_bias=False)
                self.out_channels += [base_channels]
        self._resize_cache = {}

    def construct(self, x):
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)

        intra_feat = conv2
        outputs = {}
        out = self.out1(intra_feat)
        outputs["stage1"] = out

        if self.arch_mode == "unet":
            if self.num_stage == 3:
                intra_feat = self.deconv1(conv1, intra_feat)
                outputs["stage2"] = self.out2(intra_feat)
                intra_feat = self.deconv2(conv0, intra_feat)
                outputs["stage3"] = self.out3(intra_feat)
            elif self.num_stage == 2:
                intra_feat = self.deconv1(conv1, intra_feat)
                outputs["stage2"] = self.out2(intra_feat)

        elif self.arch_mode == "fpn":
            if self.num_stage == 3:
                h, w = intra_feat.shape[2], intra_feat.shape[3]
                intra_feat = ops.interpolate(intra_feat, size=(h * 2, w * 2), mode="nearest") + self.inner1(conv1)
                outputs["stage2"] = self.out2(intra_feat)

                h, w = intra_feat.shape[2], intra_feat.shape[3]
                intra_feat = ops.interpolate(intra_feat, size=(h * 2, w * 2), mode="nearest") + self.inner2(conv0)
                outputs["stage3"] = self.out3(intra_feat)

            elif self.num_stage == 2:
                h, w = intra_feat.shape[2], intra_feat.shape[3]
                intra_feat = ops.interpolate(intra_feat, size=(h * 2, w * 2), mode="nearest") + self.inner1(conv1)
                outputs["stage2"] = self.out2(intra_feat)
        return outputs


class CostRegNet(nn.Cell):
    def __init__(self, in_channels, base_channels):
        super().__init__()
        self.conv0 = Conv3d(in_channels, base_channels, padding=1)
        self.conv1 = Conv3d(base_channels, base_channels * 2, stride=2, padding=1)
        self.conv2 = Conv3d(base_channels * 2, base_channels * 2, padding=1)
        self.conv3 = Conv3d(base_channels * 2, base_channels * 4, stride=2, padding=1)
        self.conv4 = Conv3d(base_channels * 4, base_channels * 4, padding=1)
        self.conv5 = Conv3d(base_channels * 4, base_channels * 8, stride=2, padding=1)
        self.conv6 = Conv3d(base_channels * 8, base_channels * 8, padding=1)
        self.conv7 = Deconv3d(base_channels * 8, base_channels * 4, stride=2, padding=1, output_padding=1)
        self.conv9 = Deconv3d(base_channels * 4, base_channels * 2, stride=2, padding=1, output_padding=1)
        self.conv11 = Deconv3d(base_channels * 2, base_channels * 1, stride=2, padding=1, output_padding=1)
        self.prob = nn.Conv3d(base_channels, 1, 3, stride=1, padding=1, pad_mode='pad', has_bias=False)

    def construct(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        x = conv2 + self.conv9(x)
        x = conv0 + self.conv11(x)
        x = self.prob(x)
        return x

class RefineNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvBnReLU(4, 32)
        self.conv2 = ConvBnReLU(32, 32)
        self.conv3 = ConvBnReLU(32, 32)
        self.res = ConvBnReLU(32, 1)
        self.concat = ops.Concat(axis=1)

    def construct(self, img, depth_init):
        concat = self.concat((img, depth_init))
        depth_residual = self.res(self.conv3(self.conv2(self.conv1(concat))))
        depth_refined = depth_init + depth_residual
        return depth_refined


def depth_regression(p, depth_values):
    """
    p: [B, D, H, W]
    depth_values: [B, D] 或 [B, D, 1, 1] 或 [D]
    """
    if depth_values.ndim == 2:  # [B, D] -> [B, D, 1, 1]
        B, D = depth_values.shape
        depth_values = ops.reshape(depth_values, (B, D, 1, 1))
    elif depth_values.ndim == 1:  # [D] -> [1, D, 1, 1]
        D = depth_values.shape[0]
        depth_values = ops.reshape(depth_values, (1, D, 1, 1))
    
    depth = ops.reduce_sum(p * depth_values, axis=1)
    return depth

def cas_mvsnet_loss(inputs, depth_gt_ms, mask_ms, **kwargs):
    dlossw = kwargs.get("dlossw", None)
    total_loss = ms.Tensor(0.0, dtype=ms.float32)
    smooth_l1 = nn.SmoothL1Loss(reduction='none')
    for stage_key in sorted([k for k in inputs.keys() if "stage" in k]):
        depth_est = inputs[stage_key]["depth"]
        depth_gt = depth_gt_ms[stage_key]
        mask = mask_ms[stage_key] > 0.5  # bool
        loss_map = smooth_l1(depth_est, depth_gt)  # [B,1,H,W] or [B,H,W]
        if len(loss_map.shape) == 4 and loss_map.shape[1] == 1:
            loss_map = ops.squeeze(loss_map, axis=1)
        mask_f = ops.cast(mask, ms.float32)
        loss = ops.reduce_sum(loss_map * mask_f) / (ops.reduce_sum(mask_f) + 1e-6)

        if dlossw is not None:
            stage_idx = int(stage_key.replace("stage", "")) - 1
            total_loss = total_loss + dlossw[stage_idx] * loss
        else:
            total_loss = total_loss + loss

    return total_loss

def get_cur_depth_range_samples(cur_depth, ndepth, depth_inteval_pixel, shape,
                                max_depth=192.0, min_depth=0.0):
    # shape: (B, H, W) ; cur_depth: (B, H, W)
    # clamp -> clip
    cur_depth_min = ops.clip_by_value(cur_depth - (ndepth // 2) * depth_inteval_pixel, 1e-4, 1e6)
    cur_depth_max = ops.clip_by_value(cur_depth + (ndepth // 2) * depth_inteval_pixel, 1e-4, 1e4)

    assert tuple(cur_depth.shape) == tuple(shape), f"cur_depth:{cur_depth.shape}, input shape:{shape}"

    new_interval = (cur_depth_max - cur_depth_min) / (ndepth - 1)  # (B, H, W)
    ar = mnp.arange(0, ndepth, dtype=cur_depth.dtype)              # (D,)
    ar = ops.reshape(ar, (1, -1, 1, 1))                            # (1, D, 1, 1)
    depth_range_samples = ops.expand_dims(cur_depth_min, 1) + ar * ops.expand_dims(new_interval, 1)
    return ops.clip_by_value(depth_range_samples, 1e-5, 1e6)

def get_depth_range_samples(cur_depth, ndepth, depth_inteval_pixel, dtype, shape,
                            max_depth=192.0, min_depth=0.0):
    # shape: (B,H,W)
    if len(cur_depth.shape) == 2:
        cur_depth_min = cur_depth[:, 0]
        cur_depth_max = cur_depth[:, -1]
        new_interval = (cur_depth_max - cur_depth_min) / (ndepth - 1)
        ar = mnp.arange(0, ndepth, dtype=cur_depth.dtype).reshape(1, -1)  # (1, D)
        depth_range = ops.expand_dims(cur_depth_min, 1) + ar * ops.expand_dims(new_interval, 1)  # (B,D)
        depth_range = ops.reshape(depth_range, (depth_range.shape[0], depth_range.shape[1], 1, 1))
        depth_range = ops.tile(depth_range, (1, 1, shape[1], shape[2]))
    else:
        depth_range = get_cur_depth_range_samples(cur_depth, ndepth, depth_inteval_pixel, shape, max_depth, min_depth)
    return depth_range
