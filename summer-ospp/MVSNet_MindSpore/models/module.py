
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as mnp
import numpy as np
import cv2  # 用于 grid_sample 替代

class ConvBnReLU(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, pad_mode='pad')
        self.bn = nn.BatchNorm2d(out_channels)
        # self.relu = nn.ReLU()
    def construct(self, x):
        return ops.relu(self.bn(self.conv(x)))


class ConvBn(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBn, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, pad_mode='pad', has_bias=True)
        self.bn = nn.BatchNorm2d(out_channels)

    def construct(self, x):
        return self.bn(self.conv(x))


class ConvBnReLU3D(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, pad_mode='pad', has_bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        # self.relu = nn.ReLU()
    def construct(self, x):
        return ops.relu(self.bn(self.conv(x)))


class ConvBn3D(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBn3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, pad_mode='pad', has_bias=True)
        self.bn = nn.BatchNorm3d(out_channels)

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
                               has_bias=True, dtype=mindspore.float16),
            nn.BatchNorm3d(channels * 2))

        self.dconv1 = nn.SequentialCell(
            nn.Conv3dTranspose(channels * 2, channels, kernel_size=3, padding=1, output_padding=1, stride=2,
                               has_bias=True, dtype=mindspore.float16),
            nn.BatchNorm3d(channels))

        self.redir1 = ConvBn3D(channels, channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = ConvBn3D(channels * 2, channels * 2, kernel_size=1, stride=1, pad=0)
        # self.relu = nn.ReLU()
    def construct(self, x):
        x = ops.cast(x, mindspore.float16)
        conv1 = self.conv1b(self.conv1a(x))
        conv2 = self.conv2b(self.conv2a(conv1))
        dconv2 = ops.relu(self.dconv2(conv2) + self.redir2(conv1))
        dconv1 = ops.relu(self.dconv1(dconv2) + self.redir1(x))
        dconv1 = ops.cast(dconv1, mindspore.float32)
        return dconv1

def homo_warping(src_fea, src_proj, ref_proj, depth_values):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, Ndepth]
    # out: [B, C, Ndepth, H, W]
    src_fea = ops.stop_gradient(src_fea)
    src_proj = ops.stop_gradient(src_proj)
    ref_proj = ops.stop_gradient(ref_proj)
    depth_values = ops.stop_gradient(depth_values)    
    
    
    batch, channels = src_fea.shape[0], src_fea.shape[1]
    num_depth = depth_values.shape[1]
    height, width = src_fea.shape[2], src_fea.shape[3]

    proj = ops.matmul(src_proj, ops.inverse(ref_proj))
    rot = proj[:, :3, :3]  # [B,3,3]
    trans = proj[:, :3, 3:4]  # [B,3,1]
    y, x = ops.meshgrid(mnp.arange(0, height, dtype=mindspore.float32),
                    mnp.arange(0, width, dtype=mindspore.float32))

    y, x = y.reshape(-1), x.reshape(-1)   # [H*W], 三[H*W]
    xyz = ops.stack((x, y, ops.ones_like(x)), axis=0)  # [3, H*W]
    xyz = ops.expand_dims(xyz, 0)         # [1, 3, H*W]
    xyz = ops.tile(xyz, (batch, 1, 1))        # [B, 3, H*W]
    
    # 投影计算
    # print("rot shape:", rot.shape)       # 预期 [B, 3, 3]
    # print("xyz shape:", xyz.shape)       # 预期 [B, 3, H*W]
    # print("rot dtype:", rot.dtype)       # 预期 [B, 3, 3]
    # print("xyz dtype:", xyz.dtype)       # 预期 [B, 3, H*W]
    rot_xyz = ops.matmul(rot, xyz)        # [B, 3, H*W]
    rot_depth_xyz = ops.expand_dims(rot_xyz, 2)   # [B, 3, 1, H*W]
    rot_depth_xyz = ops.tile(rot_depth_xyz, (1, 1, num_depth, 1))   # [B, 3, Ndepth, H*W]
    depth_values = ops.expand_dims(depth_values, -1)             # [B, 1, Ndepth, 1]
    rot_depth_xyz = rot_depth_xyz * ops.expand_dims(depth_values, 1)  # [B, 3, Ndepth, H*W]

    proj_xyz = rot_depth_xyz + ops.expand_dims(trans, 2)  # [B, 3, Ndepth, H*W]

    # 避免除 0
    denom = proj_xyz[:, 2:3, :, :]
    denom = ops.select(ops.equal(denom, 0.0), denom + 1e-8, denom)

    # 归一化坐标
    proj_xy = proj_xyz[:, :2, :, :] / denom   # [B, 2, Ndepth, H*W]
    proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
    proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
    
    # 组装网格 [B, Ndepth, H*W, 2]
    proj_xy = ops.stack((proj_x_normalized, proj_y_normalized), axis=-1)

    # [B, Ndepth*H, W, 2] 供 grid_sample 使用
    grid = proj_xy.view(batch, num_depth * height, width , 2)

    # 双线性采样
    warped_src_fea = ops.grid_sample(src_fea,
                                     grid,
                                     mode='bilinear',
                                     padding_mode='zeros',
                                     align_corners=True)
    warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width )

    return warped_src_fea



def homo_warping_new(src_fea, src_proj, ref_proj, depth_values):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, Ndepth] o [B, Ndepth, H, W]
    # out: [B, C, Ndepth, H, W]

    batch, channels = src_fea.shape[0], src_fea.shape[1]
    num_depth = depth_values.shape[1]
    height, width = src_fea.shape[2], src_fea.shape[3]
    # 确保 ref_proj 是 float32 类型
    ref_proj = ops.cast(ref_proj, mindspore.float32)
    src_proj = ops.cast(src_proj, mindspore.float32)
    # 计算逆矩阵并矩阵乘法
    proj = ops.matmul(src_proj, ops.inverse(ref_proj))
    rot = proj[:, :3, :3]  # [B,3,3]
    trans = proj[:, :3, 3:4]  # [B,3,1]
    # 创建网格
    batch = src_fea.shape[0]
    num_depth = depth_values.shape[1]
    y, x = ops.meshgrid(mindspore.numpy.arange(0, height, dtype=mindspore.float32),
                         mindspore.numpy.arange(0, width, dtype=mindspore.float32))
    y, x = y.reshape(-1), x.reshape(-1)
    xyz = ops.stack((x, y, ops.ones_like(x)))  # [3, H*W]
    # xyz = ops.expand_dims(xyz, 0).repeat(batch, axis=0)  # [B, 3, H*W]
    xyz = ops.expand_dims(xyz, 0)  # [1, 3, H*W]
    xyz = ops.tile(xyz, (batch, 1, 1))  # [B, 3, H*W]
    # 3D投影计算
    print("rot shape:", rot.shape)       # 预期 [B, 3, 3]
    print("xyz shape:", xyz.shape)       # 预期 [B, 3, H*W]
    rot_xyz = ops.matmul(rot, xyz)  # [B, 3, H*W]
    # rot_depth_xyz = ops.expand_dims(rot_xyz, 2).repeat(num_depth, axis=2) * depth_values.view(batch, 1, num_depth, -1)  # [B, 3, Ndepth, H*W]
    rot_depth_xyz = ops.expand_dims(rot_xyz, 2)  # [B, 3, 1, H*W]
    rot_depth_xyz = rot_depth_xyz.broadcast_to((batch, 3, num_depth, rot_xyz.shape[-1]))  # [B, 3, Ndepth, H*W]
    rot_depth_xyz = rot_depth_xyz * depth_values.view(batch, 1, num_depth, -1)
    proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
    # 处理除以零的情况
    proj_xyz_slice = proj_xyz[:, 2:3, :, :]
    proj_xyz_slice = ops.select(ops.equal(proj_xyz_slice, 0),
                               proj_xyz_slice + 1e-8,
                               proj_xyz_slice)
    proj_xyz = ops.concat([proj_xyz[:, :2, :, :], proj_xyz_slice], axis=1)
    # 归一化坐标
    proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]
    # 归一化到[-1,1]范围
    proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
    proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
    # 重组网格
    proj_xy = ops.stack((proj_x_normalized, proj_y_normalized), axis=3)  # [B, Ndepth, H*W, 2]
    grid = proj_xy.view(batch, num_depth * height, width, 2)
    # 双线性采样
    warped_src_fea = ops.grid_sample(src_fea, 
                                   grid, 
                                   mode='bilinear',
                                   padding_mode='zeros',
                                   align_corners=True)   
    return warped_src_fea

# p: probability volume [B, D, H, W]
# depth_values: discrete depth values [B, D]
def depth_regression(p, depth_values):
    depth_values = depth_values.view(*depth_values.shape, 1, 1)
    depth = ops.sum(p * depth_values, 1)

    return depth