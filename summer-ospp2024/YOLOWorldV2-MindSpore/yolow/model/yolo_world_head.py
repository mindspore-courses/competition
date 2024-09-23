# Copyright (c) Tencent Inc. All rights reserved.
import copy
import numpy as np

from typing import List, Optional, Sequence, Tuple, Union

from .layers import Conv
from .misc import (get_box_wh, make_divisible, multi_apply,  yolow_dict,
                   ms_filter_scores_and_topk, ms_scale_boxes, ms_batched_nms)
from .task_utils import MlvlPointGenerator, DistancePointBBoxCoder

import mindspore.nn as nn
import mindspore as ms
from mindspore import Tensor

import time

__all__ = (
    'YOLOWorldHeadModule',
    'YOLOWorldHead',
)

    
class ContrastiveHead(nn.Cell):
    """Contrastive Head for YOLO-World
    """

    def __init__(self, use_einsum: bool = True) -> None:
        super().__init__()
        self.bias = ms.Parameter(ms.ops.zeros([]))
        self.logit_scale = ms.Parameter(ms.ops.ones([]) * np.log(1 / 0.07))
        self.use_einsum = use_einsum

    def construct(self, x: Tensor, w: Tensor) -> Tensor:
        """Forward function of contrastive learning."""
        x = ms.ops.L2Normalize(x, axis=1)
        w = ms.ops.L2Normalize(w, axis=-1)

        if self.use_einsum:
            x = ms.ops.einsum('bchw,bkc->bkhw', x, w)
        else:
            batch, channel, height, width = x.shape
            _, k, _ = w.shape
            x = x.permute(0, 2, 3, 1)  # bchw->bhwc
            x = x.reshape(batch, -1, channel)  # bhwc->b(hw)c
            w = w.permute(0, 2, 1)  # bkc->bck
            x = ms.ops.matmul(x, w)
            x = x.reshape(batch, height, width, k)
            x = x.permute(0, 3, 1, 2)

        x = x * self.logit_scale.exp() + self.bias
        return x

    
class BNContrastiveHead(nn.Cell):
    """ Batch Norm Contrastive Head for YOLO-World
    using batch norm instead of l2-normalization
    """

    def __init__(self, embed_dims: int, use_einsum: bool = True) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims, momentum=0.97, eps=0.001)
        self.bias = ms.Parameter(ms.ops.zeros([]))
        # use -1.0 is more stable
        self.logit_scale = ms.Parameter(-1.0 * ms.ops.ones([]))
        self.use_einsum = use_einsum

    def construct(self, x: Tensor, w: Tensor) -> Tensor:
        """Forward function of contrastive learning."""
        x = self.norm(x)
        l2_normalize = ms.ops.L2Normalize(axis=-1)
        w = l2_normalize(w)

        
        if self.use_einsum:
            x = ms.ops.einsum('bchw,bkc->bkhw', x, w)
        else:
            batch, channel, height, width = x.shape
            _, k, _ = w.shape
            x = x.permute(0, 2, 3, 1)  # bchw->bhwc
            x = x.reshape(batch, -1, channel)  # bhwc->b(hw)c
            w = w.permute(0, 2, 1)  # bkc->bck
            x = ms.ops.matmul(x, w)
            x = x.reshape(batch, height, width, k)
            x = x.permute(0, 3, 1, 2)

        x = x * self.logit_scale.exp() + self.bias
        return x


        
class YOLOWorldHeadModule(nn.Cell):
    """Head Module for YOLO-World
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: Union[int, Sequence],
                 embed_dims: int,
                 use_bn_head: bool = True,
                 use_einsum: bool = True,
                 freeze_all: bool = False,
                 widen_factor: float = 1.0,
                 num_base_priors: int = 1,
                 featmap_strides: Sequence[int] = (8, 16, 32),
                 reg_max: int = 16,
                 with_norm: bool = True,
                 with_activation: bool = True) -> None:
        super().__init__()

        self.embed_dims = embed_dims
        self.use_bn_head = use_bn_head
        self.use_einsum = use_einsum
        self.freeze_all = freeze_all
        self.num_classes = num_classes
        self.featmap_strides = featmap_strides
        self.num_levels = len(self.featmap_strides)
        self.num_base_priors = num_base_priors
        self.with_norm = with_norm
        self.with_activation = with_activation
        self.in_channels = in_channels
        self.reg_max = reg_max

        in_channels = []
        for channel in self.in_channels:
            channel = make_divisible(channel, widen_factor)
            in_channels.append(channel)
        self.in_channels = in_channels

        self._init_layers()

# TODO: init_weights in mindspore 
    # def init_weights(self, prior_prob=0.01):
    #     """Initialize the weight and bias of PPYOLOE head."""
    #     for reg_pred, cls_pred, cls_contrast, stride in zip(self.reg_preds, self.cls_preds, self.cls_contrasts,
    #                                                         self.featmap_strides):
    #         reg_pred[-1].bias.data[:] = 1.0  # box
    #         cls_pred[-1].bias.data[:] = 0.0  # reset bias
    #         if hasattr(cls_contrast, 'bias'):
    #             nn.init.constant_(cls_contrast.bias.data, math.log(5 / self.num_classes / (640 / stride)**2))

    def _init_layers(self) -> None:
        """initialize conv layers in YOLOv8 head."""
        # Init decouple head
        self.cls_preds = nn.CellList(auto_prefix=False)
        self.reg_preds = nn.CellList(auto_prefix=False)
        self.cls_contrasts = nn.CellList(auto_prefix=False)  # 避免自动获取前缀命名


        reg_out_channels = max((16, self.in_channels[0] // 4, self.reg_max * 4))
        cls_out_channels = max(self.in_channels[0], self.num_classes)

        for i in range(self.num_levels):
            self.reg_preds.append(
                nn.SequentialCell(
                    Conv(
                        in_channels=self.in_channels[i],
                        out_channels=reg_out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        with_norm=self.with_norm,
                        with_activation=self.with_activation),
                    Conv(
                        in_channels=reg_out_channels,
                        out_channels=reg_out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        with_norm=self.with_norm,
                        with_activation=self.with_activation),
                    nn.Conv2d(in_channels=reg_out_channels, out_channels=4 * self.reg_max, kernel_size=1, has_bias=True, pad_mode='valid')))
            
            self.cls_preds.append(
                nn.SequentialCell(
                    Conv(
                        in_channels=self.in_channels[i],
                        out_channels=cls_out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        with_norm=self.with_norm,
                        with_activation=self.with_activation),
                    Conv(
                        in_channels=cls_out_channels,
                        out_channels=cls_out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        with_norm=self.with_norm,
                        with_activation=self.with_activation),
                    nn.Conv2d(in_channels=cls_out_channels, out_channels=self.embed_dims, kernel_size=1, has_bias=True, pad_mode='valid')))
            if self.use_bn_head:
                self.cls_contrasts.append(BNContrastiveHead(self.embed_dims, use_einsum=self.use_einsum))
            else:
                self.cls_contrasts.append(ContrastiveHead(self.embed_dims, use_einsum=self.use_einsum))

        
        # proj = ms.ops.arange(self.reg_max, dtype=torch.float)
        # self.register_buffer('proj', proj, persistent=False)
        self.proj = ms.Parameter(Tensor(ms.ops.arange(self.reg_max, dtype=ms.float32)), requires_grad=False)
        

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_all:
            self._freeze_all()

    def construct(self, img_feats: Tuple[Tensor], txt_feats: Tensor) -> Tuple[List]:
        """Forward features from the upstream network."""
        assert len(img_feats) == self.num_levels
        txt_feats = [txt_feats for _ in range(self.num_levels)]

        res = multi_apply(self.forward_single, img_feats, txt_feats, self.cls_preds, self.reg_preds,
                           self.cls_contrasts)

        return res

    def forward_single(self, img_feat: Tensor, txt_feat: Tensor, cls_pred: nn.CellList, reg_pred: nn.CellList,
                       cls_contrast: nn.CellList) -> Tuple:
        """Forward feature of a single scale level."""
        b, _, h, w = img_feat.shape
        cls_embed = cls_pred(img_feat)
        cls_logit = cls_contrast(cls_embed, txt_feat)
        bbox_dist_preds = reg_pred(img_feat)
        if self.reg_max > 1:
            bbox_dist_preds = bbox_dist_preds.reshape([-1, 4, self.reg_max, h * w]).permute(0, 3, 1, 2)

            # TODO: The get_flops script cannot handle the situation of
            #  matmul, and needs to be fixed later            
            bbox_preds = ms.ops.matmul(ms.ops.softmax(bbox_dist_preds, axis=3),self.proj.unsqueeze(1)).squeeze(-1)
            bbox_preds = ms.ops.swapaxes(bbox_preds, 1, 2).reshape(b, -1, h, w)
        else:
            bbox_preds = bbox_dist_preds
        if self.training:
            return cls_logit, bbox_preds, bbox_dist_preds
        else:
            return cls_logit, bbox_preds


class YOLOWorldHead(nn.Cell):
    """YOLO-World Head

    - loss(): forward() -> loss_by_feat()
    - predict(): forward() -> predict_by_feat()
    - loss_and_predict(): forward() -> loss_by_feat() -> predict_by_feat()
    """

    def __init__(self, head_module: nn.Cell, test_cfg: Optional[dict] = None) -> None:
        super().__init__()

        self.head_module = head_module
        self.num_classes = self.head_module.num_classes
        self.featmap_strides = self.head_module.featmap_strides
        self.num_levels = len(self.featmap_strides)

        self.test_cfg = test_cfg


        # init task_utils
        self.prior_generator = MlvlPointGenerator(offset=0.5, strides=[8, 16, 32])
        self.bbox_coder = DistancePointBBoxCoder()
        self.num_base_priors = self.prior_generator.num_base_priors[0]
        # TODO later 0722
        self.featmap_sizes = [ms.numpy.empty([1])] * self.num_levels

        self.prior_match_thr = 4.0
        self.near_neighbor_thr = 0.5
        self.obj_level_weights = [4.0, 1.0, 0.4]
        self.ignore_iof_thr = -1.0

        # Add common attributes to reduce calculation
        self.featmap_sizes_train = None
        self.num_level_priors = None
        self.flatten_priors_train = None
        self.stride_tensor = None


    def construct(self,
                img_feats: Tuple[Tensor],
                txt_feats: Tensor,
                batch_data_samples: Union[list, dict],
                rescale: bool = False) -> list:
        """Perform forward propagation of the detection head and predict
        detection results on the features of the upstream network.
        """
        batch_img_metas = [
            data_samples['img_metas'] for data_samples in batch_data_samples  # changed `.metainfo` to ['img_metas']
        ]
        outs = self.head_module(img_feats, txt_feats)

        predictions = self.predict_by_feat(*outs, batch_img_metas=batch_img_metas, rescale=rescale)
        return predictions


    def predict(self,
                img_feats: Tuple[Tensor],
                txt_feats: Tensor,
                batch_data_samples: Union[list, dict],
                rescale: bool = False) -> list:
        """Perform forward propagation of the detection head and predict
        detection results on the features of the upstream network.
        """
        batch_img_metas = [
            data_samples['img_metas'] for data_samples in batch_data_samples  # changed `.metainfo` to ['img_metas']
        ]
        outs = self(img_feats, txt_feats)

        predictions = self.predict_by_feat(*outs, batch_img_metas=batch_img_metas, rescale=rescale)
        return predictions

    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        objectnesses: Optional[List[Tensor]] = None,
                        batch_img_metas: Optional[List[dict]] = None,
                        cfg: Optional[dict] = None,
                        rescale: bool = True,
                        with_nms: bool = True) -> List:
        """Transform a batch of output features extracted by the head into
        bbox results.
        """
        assert len(cls_scores) == len(bbox_preds)
        if objectnesses is None:
            with_objectnesses = False
        else:
            with_objectnesses = True
            assert len(cls_scores) == len(objectnesses)

        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)

        multi_label = cfg.multi_label
        multi_label &= self.num_classes > 1
        cfg.multi_label = multi_label

        num_imgs = len(batch_img_metas)
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        
        # If the shape does not change, use the previous mlvl_priors
        if featmap_sizes != self.featmap_sizes:
            self.mlvl_priors = self.prior_generator.grid_priors(
                featmap_sizes, dtype=cls_scores[0].dtype)
            # featmap_sizes = [ms.ops.full_like(f) for f in featmap_sizes]
            self.featmap_sizes = featmap_sizes
        
        flatten_priors = ms.ops.cat(self.mlvl_priors)
        mlvl_strides = []
        # TODO: mindspore 获取Tensor的shape 比如 [160, 160]，统计numel的时候返回2，不像torch 返回160*160
        for featmap_size, stride in zip(featmap_sizes, self.featmap_strides):
            tmp = ms.ops.full((featmap_size[0]*featmap_size[1] * self.num_base_priors, ), stride)    
            mlvl_strides.append(tmp)
        # mlvl_strides = [
        #     ms.ops.full((ms.ops.numel(featmap_size) * self.num_base_priors, ), stride)
        #     # flatten_priors.new_full((featmap_size.numel() * self.num_base_priors, ), stride)
        #     for featmap_size, stride in zip(featmap_sizes, self.featmap_strides)
        # ]
        
        flatten_stride = ms.ops.cat(mlvl_strides)

        # flatten cls_scores, bbox_preds and objectness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.num_classes) for cls_score in cls_scores
        ]
        flatten_bbox_preds = [bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4) for bbox_pred in bbox_preds]

        flatten_cls_scores = ms.ops.cat(flatten_cls_scores, axis=1).sigmoid()
        flatten_bbox_preds = ms.ops.cat(flatten_bbox_preds, axis=1)
        
        flatten_decoded_bboxes = self.bbox_coder.decode(flatten_priors[None], flatten_bbox_preds, flatten_stride)

        if with_objectnesses:
            flatten_objectness = [objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1) for objectness in objectnesses]
            flatten_objectness = ms.ops.cat(flatten_objectness, axis=1).sigmoid()
        else:
            flatten_objectness = [None for _ in range(num_imgs)]
        results_list = []
        
        for (bboxes, scores, objectness, img_meta) in zip(flatten_decoded_bboxes, flatten_cls_scores,
                                                          flatten_objectness, batch_img_metas):
            ori_shape = img_meta['ori_shape']
            scale_factor = img_meta['scale_factor']
            if 'pad_param' in img_meta:
                pad_param = img_meta['pad_param']
            else:
                pad_param = None

            score_thr = cfg.get('score_thr', -1)
            # yolox_style does not require the following operations
            if objectness is not None and score_thr > 0 and not cfg.get('yolox_style', False):
                conf_inds = objectness > score_thr
                bboxes = bboxes[conf_inds, :]
                scores = scores[conf_inds, :]
                objectness = objectness[conf_inds]

            if objectness is not None:
                # conf = obj_conf * cls_conf
                scores *= objectness[:, None]

            if scores.shape[0] == 0:
                empty_results = yolow_dict()
                empty_results.bboxes = bboxes
                empty_results.scores = scores[:, 0]
                empty_results.labels = scores[:, 0].int()
                results_list.append(empty_results)
                continue

            nms_pre = cfg.get('nms_pre', 10000)
            
            if cfg.multi_label is False:
                scores, labels = scores.max(1, keepdim=True)
                scores, _, keep_idxs, results = ms_filter_scores_and_topk(
                    scores, score_thr, nms_pre, results=dict(labels=labels[:, 0]))
                labels = results['labels']
            else:
                scores, labels, keep_idxs, _ = ms_filter_scores_and_topk(scores, score_thr, nms_pre)
            
            
            results = yolow_dict(scores=scores, labels=labels, bboxes=bboxes[keep_idxs])

            if rescale:
                if pad_param is not None:
                    results.bboxes -= ms.Tensor([pad_param[2], pad_param[0], pad_param[2], pad_param[0]])
                
                results.bboxes /= ms.Tensor(scale_factor).repeat(2).unsqueeze(0)

            if cfg.get('yolox_style', False):
                # do not need max_per_img
                cfg.max_per_img = len(results)
            
            results = self._bbox_post_process(
                results=results, cfg=cfg, rescale=False, with_nms=with_nms, img_meta=img_meta)

            results.bboxes[:, 0::2] = ms.ops.clamp(results.bboxes[:, 0::2], 0, ori_shape[1])
            results.bboxes[:, 1::2] = ms.ops.clamp(results.bboxes[:, 1::2], 0, ori_shape[0])

            results_list.append(results)
        
        return results_list

    def _bbox_post_process(self,
                           results: dict,
                           cfg: dict,
                           rescale: bool = False,
                           with_nms: bool = True,
                           img_meta: Optional[dict] = None) -> dict:
        """bbox post-processing method.

        The boxes would be rescaled to the original image scale and do
        the nms operation. Usually `with_nms` is False is used for aug test.
        """
        if rescale:
            assert img_meta.get('scale_factor') is not None
            scale_factor = [1 / s for s in img_meta['scale_factor']]
            results.bboxes = ms_scale_boxes(results.bboxes, scale_factor)

        if hasattr(results, 'score_factors'):
            # TODO: Add sqrt operation in order to be consistent with
            #  the paper.
            score_factors = results.pop('score_factors')
            results.scores = results.scores * score_factors

        # filter small size bboxes
        if cfg.get('min_bbox_size', -1) >= 0:
            w, h = get_box_wh(results.bboxes)
            valid_mask = (w > cfg.min_bbox_size) & (h > cfg.min_bbox_size)
            if not valid_mask.all():
                results = results[valid_mask]

        # TODO: deal with `with_nms` and `nms_cfg=None` in test_cfg
        
        if with_nms and results.bboxes.numel() > 0:
            # bboxes = get_box_tensor(results.bboxes)
            bboxes = results.bboxes
            assert isinstance(bboxes, Tensor)
            # start = time.perf_counter()
            det_bboxes, keep_idxs = ms_batched_nms(bboxes, results.scores, results.labels, cfg.nms)
            # topk_time = time.perf_counter()
            # print(f"ms_batched_nms time: {topk_time - start}")
            # results = results[keep_idxs]
            for k in results.keys():
                results[k] = results[k][keep_idxs]
            # some nms would reweight the score, such as softnms
            results.scores = det_bboxes[:, -1]
            for k in results.keys():
                results[k] = results[k][:cfg.max_per_img]

        return results