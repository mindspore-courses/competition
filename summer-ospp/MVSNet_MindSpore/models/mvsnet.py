
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

from .module import *


class FeatureNet(nn.Cell):
    def __init__(self):
        super(FeatureNet, self).__init__()
        self.inplanes = 32

        self.conv0 = ConvBnReLU(3, 8, 3, 1, 1)
        self.conv1 = ConvBnReLU(8, 8, 3, 1, 1)

        self.conv2 = ConvBnReLU(8, 16, 5, 2, 2)
        self.conv3 = ConvBnReLU(16, 16, 3, 1, 1)
        self.conv4 = ConvBnReLU(16, 16, 3, 1, 1)

        self.conv5 = ConvBnReLU(16, 32, 5, 2, 2)
        self.conv6 = ConvBnReLU(32, 32, 3, 1, 1)
        # self.feature = nn.Conv2d(32, 32, 3, 1, 1)
        self.feature = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, pad_mode='pad', has_bias=True)

    def construct(self, x):
        x = self.conv1(self.conv0(x))
        x = self.conv4(self.conv3(self.conv2(x)))
        x = self.feature(self.conv6(self.conv5(x)))
        return x


class CostRegNet(nn.Cell):
    def __init__(self):
        super(CostRegNet, self).__init__()
        self.conv0 = ConvBnReLU3D(32, 8)

        self.conv1 = ConvBnReLU3D(8, 16, stride=2)
        self.conv2 = ConvBnReLU3D(16, 16)

        self.conv3 = ConvBnReLU3D(16, 32, stride=2)
        self.conv4 = ConvBnReLU3D(32, 32)

        self.conv5 = ConvBnReLU3D(32, 64, stride=2)
        self.conv6 = ConvBnReLU3D(64, 64)

        self.conv7 = nn.SequentialCell(
            nn.Conv3dTranspose(64, 32, kernel_size=3, padding=1, output_padding=1,pad_mode="pad",  stride=2,has_bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU())

        self.conv9 = nn.SequentialCell(
            nn.Conv3dTranspose(32, 16, kernel_size=3, padding=1, output_padding=1, pad_mode="pad", stride=2,has_bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU())

        self.conv11 = nn.SequentialCell(
            nn.Conv3dTranspose(16, 8, kernel_size=3, padding=1, output_padding=1, pad_mode="pad", stride=2,has_bias=False),
            nn.BatchNorm3d(8),
            nn.ReLU())

        self.prob = nn.Conv3d(8, 1, 3, stride=1, padding=1, pad_mode="pad",has_bias=True)

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
        super(RefineNet, self).__init__()
        self.conv1 = ConvBnReLU(4, 32)
        self.conv2 = ConvBnReLU(32, 32)
        self.conv3 = ConvBnReLU(32, 32)
        self.res = ConvBnReLU(32, 1)

    def construct(self, img, depth_init):
        concat = ops.concat((img, depth_init), axis=1)
        depth_residual = self.res(self.conv3(self.conv2(self.conv1(concat))))
        depth_refined = depth_init + depth_residual
        return depth_refined


class MVSNet(nn.Cell):
    def __init__(self, refine=True):
        super(MVSNet, self).__init__()
        self.refine = refine

        self.feature = FeatureNet()
        self.cost_regularization = CostRegNet()
        if self.refine:
            self.refine_network = RefineNet()
        self.softmax = ops.Softmax(axis=1)
        self.pad = ops.Pad(((0, 0), (0, 0), (1, 2), (0, 0), (0, 0)))  # for avg_pool3d
        self.avgpool3d = ops.AvgPool3D(kernel_size=(4, 1, 1), strides=1, pad_mode="valid")
    def construct(self, imgs, proj_matrices, depth_values):
        # imgs: [B, V, C, H, W]
        # proj_matrices: [B, V, 4, 4]
        imgs = ops.unstack(imgs, axis=1)
        proj_matrices = ops.unstack(proj_matrices, axis=1)
        assert len(imgs) == len(proj_matrices), "Different number of images and projection matrices"
        img_height, img_width = imgs[0].shape[2], imgs[0].shape[3]
        num_depth = depth_values.shape[1]
        num_views = len(imgs)

        # step 1. feature extraction
        # in: images; out: 32-channel feature maps
        features = [self.feature(img) for img in imgs]
        ref_feature, src_features = features[0], features[1:]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]

        # step 2. differentiable homograph, build cost volume
        ref_volume = ops.tile(ops.expand_dims(ref_feature, 2), (1, 1, num_depth, 1, 1))
        volume_sum = ref_volume
        volume_sq_sum = ops.pow(ref_volume, 2)

        for src_fea, src_proj in zip(src_features, src_projs):
            # warpped features
            warped_volume = homo_warping(src_fea, src_proj, ref_proj, depth_values)
            volume_sum = volume_sum + warped_volume
            volume_sq_sum = volume_sq_sum + ops.pow(warped_volume, 2)
        # aggregate multiple feature volumes by variance
        volume_variance = volume_sq_sum / num_views - ops.pow(volume_sum / num_views, 2)

        # step 3. cost volume regularization
        cost_reg = self.cost_regularization(volume_variance)
        cost_reg = ops.squeeze(cost_reg, 1)
        prob_volume = self.softmax(cost_reg)
        depth = depth_regression(prob_volume, depth_values=depth_values)
            # photometric confidence
        
        prob_volume_ng = ops.stop_gradient(prob_volume)
        prob_volume_sum4 = 4 * self.avgpool3d(self.pad(ops.expand_dims(prob_volume_ng, 1))).squeeze(1)
        depth_index = depth_regression(prob_volume_ng, depth_values=ops.arange(num_depth, dtype=mindspore.float32)).astype(mindspore.int32)

        photometric_confidence = ops.gather_elements(prob_volume_sum4, 1,ops.expand_dims(depth_index, 1)).squeeze(1)


        # step 4. depth map refinement
        if not self.refine:
            return {"depth": depth, "photometric_confidence": photometric_confidence}
        else:
            refined_depth = self.refine_network(ops.concat((imgs[0], depth), axis=1))
            return {"depth": depth, "refined_depth": refined_depth, "photometric_confidence": photometric_confidence}


def mvsnet_loss(depth_est, depth_gt, mask):
    # mask: [B, H, W]  (0/1 或 float)
    mask = mask > 0.5  # BoolTensor

    # 取出 mask 位置的元素
    depth_est_valid = ops.masked_select(depth_est, mask)
    depth_gt_valid = ops.masked_select(depth_gt, mask)

    # 防止空 mask
    if depth_est_valid.size == 0:
        return ms.Tensor(0.0, ms.float32)

    # Smooth L1 Loss
    loss_fn = nn.SmoothL1Loss(reduction="mean")
    loss = loss_fn(depth_est_valid, depth_gt_valid)
    return loss