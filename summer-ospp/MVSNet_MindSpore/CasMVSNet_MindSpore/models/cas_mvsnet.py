import mindspore
import mindspore.nn as nn
import mindspore.ops as ops

from .module import *
Align_Corners_Range = False


class DepthNet(nn.Cell):
    def __init__(self):
        super(DepthNet, self).__init__()
        self.softmax = ops.Softmax(axis=1)

    def construct(self, features, proj_matrices, depth_values, num_depth, cost_regularization, prob_volume_init=None):
        proj_matrices = ops.unstack(proj_matrices, axis=1)
        num_views = len(features)
        assert len(features) == num_views, "Different number of images and projection matrices"
        assert depth_values.shape[1] == num_depth, f"depth_values.shape[1]:{depth_values.shape[1]}  num_depth:{num_depth}"

        ref_feature, src_features = features[0], features[1:]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]

        # step 2. cost volume
        ref_volume = ops.tile(ops.expand_dims(ref_feature, 2), (1, 1, num_depth, 1, 1))
        volume_sum = ref_volume
        volume_sq_sum = ref_volume ** 2

        for src_fea, src_proj in zip(src_features, src_projs):
            src_proj_new = src_proj[:, 0]
            src_proj_rot = src_proj[:, 1, :3, :3]
            src_proj_trans = src_proj[:, 0, :3, :4]
            src_proj_new[:, :3, :4] = ops.matmul(src_proj_rot, src_proj_trans)
            ref_proj_new = ref_proj[:, 0]
            ref_proj_rot = ref_proj[:, 1, :3, :3]
            ref_proj_trans = ref_proj[:, 0, :3, :4]
            ref_proj_new[:, :3, :4] = ops.matmul(ref_proj_rot, ref_proj_trans)
            warped_volume = homo_warping(src_fea, src_proj_new, ref_proj_new, depth_values)

            volume_sum = volume_sum + warped_volume
            volume_sq_sum = volume_sq_sum + warped_volume ** 2

        volume_variance = volume_sq_sum / num_views - (volume_sum / num_views) ** 2

        cost_reg = cost_regularization(volume_variance)
        prob_volume_pre = ops.squeeze(cost_reg, 1)

        if prob_volume_init is not None:
            prob_volume_pre += prob_volume_init

        prob_volume = self.softmax(prob_volume_pre)
        depth = depth_regression(prob_volume, depth_values=depth_values)
        x = ops.expand_dims(prob_volume, 1)
        pad_op = ops.Pad(((0,0),  (0,0), (1,2),  (0,0),  (0,0)))
        x = pad_op(x)
        x = ops.avg_pool3d(x, kernel_size=(4,1,1), stride=1, padding=0)
        prob_volume_sum4 = 4 * ops.squeeze(x, 1)
        depth_index = depth_regression(prob_volume,
                                       depth_values=ops.arange(num_depth, dtype=mindspore.float32)).astype(mindspore.int32)
        depth_index = ops.clamp(depth_index, 0, num_depth - 1)
        photometric_confidence = ops.gather_elements(prob_volume_sum4, 1, ops.expand_dims(depth_index, 1)).squeeze(1)

        return {"depth": depth, "photometric_confidence": photometric_confidence}


class CascadeMVSNet(nn.Cell):
    def __init__(self, refine=False, ndepths=[48, 32, 8], depth_interals_ratio=[4, 2, 1],
                 share_cr=False, grad_method="detach", arch_mode="fpn", cr_base_chs=[8, 8, 8]):
        super(CascadeMVSNet, self).__init__()
        self.refine = refine
        self.share_cr = share_cr
        self.ndepths = ndepths
        self.depth_interals_ratio = depth_interals_ratio
        self.grad_method = grad_method
        self.arch_mode = arch_mode
        self.cr_base_chs = cr_base_chs
        self.num_stage = len(ndepths)

        self.stage_infos = {
            "stage1": {"scale": 4.0},
            "stage2": {"scale": 2.0},
            "stage3": {"scale": 1.0}
        }

        self.feature = FeatureNet(base_channels=8, stride=4, num_stage=self.num_stage, arch_mode=self.arch_mode)
        if self.share_cr:
            self.cost_regularization = CostRegNet(in_channels=self.feature.out_channels, base_channels=8)
        else:
            self.cost_regularization = nn.CellList([CostRegNet(in_channels=self.feature.out_channels[i],
                                                               base_channels=self.cr_base_chs[i])
                                                    for i in range(self.num_stage)])
        if self.refine:
            self.refine_network = RefineNet()
        self.DepthNet = DepthNet()

    def construct(self, imgs, stage1_proj, stage2_proj, stage3_proj, depth_values):
        depth_min = float(depth_values[0, 0].asnumpy())
        depth_max = float(depth_values[0, -1].asnumpy())
        depth_interval = (depth_max - depth_min) / depth_values.shape[1]

        # step 1. feature extraction
        features = []
        for nview_idx in range(imgs.shape[1]):  # imgs (B, N, C, H, W)
            img = imgs[:, nview_idx]
            features.append(self.feature(img))

        proj_matrices = {"stage1": stage1_proj, "stage2": stage2_proj, "stage3": stage3_proj}

        outputs = {}
        depth, cur_depth = None, None
        for stage_idx in range(self.num_stage):
            features_stage = [feat[f"stage{stage_idx + 1}"] for feat in features]
            proj_matrices_stage = proj_matrices[f"stage{stage_idx + 1}"]
            stage_scale = self.stage_infos[f"stage{stage_idx + 1}"]["scale"]

            if depth is not None:
                cur_depth = ops.stop_gradient(depth) if self.grad_method == "detach" else depth
                cur_depth = ops.interpolate(cur_depth.expand_dims(1),
                                            size=(img.shape[2], img.shape[3]),
                                            mode="bilinear",
                                            align_corners=Align_Corners_Range).squeeze(1)
            else:
                cur_depth = depth_values

            depth_range_samples = get_depth_range_samples(cur_depth=cur_depth,
                                                          ndepth=self.ndepths[stage_idx],
                                                          depth_inteval_pixel=self.depth_interals_ratio[stage_idx] * depth_interval,
                                                          dtype=img.dtype,
                                                          shape=(img.shape[0], img.shape[2], img.shape[3]),
                                                          max_depth=depth_max,
                                                          min_depth=depth_min)

            depth_range_samples = ops.interpolate(depth_range_samples.expand_dims(1),
                                                  size=(self.ndepths[stage_idx],
                                                         img.shape[2] // int(stage_scale),
                                                         img.shape[3] // int(stage_scale)),
                                                  mode="trilinear",
                                                  align_corners=Align_Corners_Range).squeeze(1)

            outputs_stage = self.DepthNet(features_stage, proj_matrices_stage,
                                          depth_values=depth_range_samples,
                                          num_depth=self.ndepths[stage_idx],
                                          cost_regularization=self.cost_regularization if self.share_cr
                                          else self.cost_regularization[stage_idx])

            depth = outputs_stage['depth']
            outputs[f"stage{stage_idx + 1}"] = outputs_stage
            outputs.update(outputs_stage)

        if self.refine:
            refined_depth = self.refine_network(ops.concat((imgs[:, 0], depth), axis=1))
            outputs["refined_depth"] = refined_depth

        return outputs


if __name__ == "__main__":
    import mindspore
    from mindspore import context
    from mindspore.dataset import GeneratorDataset
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")  # 或 "Ascend"/"CPU"
    from datasets import MVSDataset
    DTU_TRAINING = "/media/outbreak/68E1-B517/Dataset/DTU_ZIP/dtu_training/mvs_training/dtu_training"
    dataset = MVSDataset(
        datapath=DTU_TRAINING,
        listfile="lists/dtu/train.txt",
        mode="train",
        nviews=5,
        ndepths=192,
        interval_scale=1.06
    )

    minds_dataset = GeneratorDataset(
        dataset,
        column_names=[
            "imgs", "stage1_proj", "stage2_proj", "stage3_proj",
            "stage1_depth", "stage2_depth", "stage3_depth",
            "stage1_mask", "stage2_mask", "stage3_mask",
            "depth_values", "scanid", "viewid"
        ],
        shuffle=True
    ).batch(batch_size=1)
    iterator = minds_dataset.create_dict_iterator()
    item = next(iterator)

    print("数据检查:")
    for k, v in item.items():
        if hasattr(v, "shape"):
            print(f"{k}: {v.shape}")
        else:
            print(f"{k}: {v}")

    model = CascadeMVSNet(refine=False, ndepths=[48, 32, 8], depth_interals_ratio=[4, 2, 1])

    outputs = model(item["imgs"],
                    item["stage1_proj"],
                    item["stage2_proj"],
                    item["stage3_proj"],
                    item["depth_values"])

    print("\n模型输出检查:")
    for k, v in outputs.items():
        if hasattr(v, "shape"):
            print(f"{k}: {v.shape}")
        elif isinstance(v, dict):
            print(f"{k}: dict")
            for kk, vv in v.items():
                print(f"  {kk}: {vv.shape if hasattr(vv, 'shape') else type(vv)}")
        else:
            print(f"{k}: {type(v)}")
