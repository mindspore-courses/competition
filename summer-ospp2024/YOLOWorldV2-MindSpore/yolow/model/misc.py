# Copyright (c) OpenMMLab. All rights reserved.
# Apache-2.0 license
import math
import numpy as np

from collections import abc
from functools import partial

from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union


import mindspore.nn as nn
import mindspore as ms
from mindspore import Tensor

__all__ = (  # TODO remove useless ones
    'get_world_size',
    'get_rank',
    'get_dist_info',
    'yolow_dict',
    'gt_instances_preprocess',
    'is_seq_of',
    'is_list_of',
    'stack_batch',
    'make_divisible',
    'make_round',
    'multi_apply',
    'unpack_gt_instances',
    'filter_scores_and_topk',
    'get_prior_xy_info',
    'scale_boxes',
    'get_box_wh',
    'nms',
    'batched_nms',
    'revert_sync_batchnorm',
)


class yolow_dict(dict):

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as e:
            raise AttributeError from e

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__



def is_seq_of(seq: Any, expected_type: Union[Type, tuple], seq_type: Type = None) -> bool:
    """Check whether it is a sequence of some type.
    """
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True


def is_list_of(seq, expected_type):
    """Check whether it is a list of some type.
    """
    return is_seq_of(seq, expected_type, seq_type=list)



def make_divisible(x: float, widen_factor: float = 1.0, divisor: int = 8) -> int:
    """Make sure that x*widen_factor is divisible by divisor."""
    return math.ceil(x * widen_factor / divisor) * divisor


def make_round(x: float, deepen_factor: float = 1.0) -> int:
    """Make sure that x*deepen_factor becomes an integer not less than 1."""
    return max(round(x * deepen_factor), 1) if x > 1 else x


def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def unpack_gt_instances(batch_data_samples: list) -> tuple:
    """Unpack ``gt_instances``, ``gt_instances_ignore`` and ``img_metas`` based
    on ``batch_data_samples``
    """
    batch_gt_instances = []
    batch_gt_instances_ignore = []
    batch_img_metas = []
    for data_sample in batch_data_samples:
        batch_img_metas.append(data_sample.metainfo)
        batch_gt_instances.append(data_sample.gt_instances)
        if 'ignored_instances' in data_sample:
            batch_gt_instances_ignore.append(data_sample.ignored_instances)
        else:
            batch_gt_instances_ignore.append(None)
    return batch_gt_instances, batch_gt_instances_ignore, batch_img_metas


def ms_filter_scores_and_topk(scores, score_thr, topk, results=None):
    """Filter results using score threshold and topk candidates.
    """
    valid_mask = scores > score_thr
    scores = scores[valid_mask]
    valid_idxs = ms.ops.nonzero(valid_mask)

    num_topk = min(topk, valid_idxs.shape[0])
    # torch.sort is actually faster than .topk (at least on GPUs)
    scores, idxs = scores.sort(descending=True)
    scores = scores[:num_topk]
    topk_idxs = valid_idxs[idxs[:num_topk]]
    
    keep_idxs, labels = topk_idxs.unbind(dim=1)

    filtered_results = None
    if results is not None:
        if isinstance(results, dict):
            filtered_results = {k: v[keep_idxs] for k, v in results.items()}
        elif isinstance(results, list):
            filtered_results = [result[keep_idxs] for result in results]
        elif isinstance(results, ms.Tensor):
            filtered_results = results[keep_idxs]
        else:
            raise NotImplementedError(f'Only supports dict or list or Tensor, '
                                      f'but get {type(results)}.')
    return scores, labels, keep_idxs, filtered_results


def get_prior_xy_info(index: int, num_base_priors: int, featmap_sizes: int) -> Tuple[int, int, int]:
    """Get prior index and xy index in feature map by flatten index."""
    _, featmap_w = featmap_sizes
    priors = index % num_base_priors
    xy_index = index // num_base_priors
    grid_y = xy_index // featmap_w
    grid_x = xy_index % featmap_w
    return priors, grid_x, grid_y

    
def ms_scale_boxes(boxes: Union[Tensor, dict], scale_factor: Tuple[float, float]) -> Union[Tensor, dict]:
    """Scale boxes with type of tensor or box type.
    """
    if isinstance(boxes, dict):
        boxes.rescale_(scale_factor)
        return boxes
    else:
        # Tensor boxes will be treated as horizontal boxes
        repeat_num = int(boxes.size(-1) / 2)
        scale_factor = ms.Tensor(scale_factor).repeat((1, repeat_num))
        return boxes * scale_factor


def get_box_wh(boxes: Union[Tensor, dict]) -> Tuple[Tensor, Tensor]:
    """Get the width and height of boxes with type of tensor or box type.
    """
    if isinstance(boxes, dict):
        w = boxes.widths
        h = boxes.heights
    else:
        # Tensor boxes will be treated as horizontal boxes by defaults
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
    return w, h

    
class NMSop(nn.Cell):

    def __init__(self, iou_threshold):
        super().__init__(iou_threshold)
        self.nms_func = ms.ops.NMSWithMask(iou_threshold)

    # @staticmethod
    def construct(self, bboxes: Tensor, scores: Tensor, iou_threshold: float, offset: int, score_threshold: float, max_num: int) -> Tensor:
        is_filtering_by_score = score_threshold > 0
        if is_filtering_by_score:
            valid_mask = scores > score_threshold
            bboxes, scores = bboxes[valid_mask], scores[valid_mask]
            valid_inds = ms.ops.nonzero(valid_mask).squeeze(axis=1)

        # inds = ext_module.nms(
        #     bboxes, scores, iou_threshold=float(iou_threshold), offset=offset)
        # inds = box_ops.batched_nms(bboxes.float(), scores, torch.ones(bboxes.size(0)), iou_threshold)
        
        box_with_score = ms.ops.concat((bboxes, scores.unsqueeze(-1)), axis=1)
        
        o_bboxes,indexs,selected_mask = self.nms_func(box_with_score)

        inds = indexs[selected_mask]

        
        if max_num > 0:
            inds = inds[:max_num]
        if is_filtering_by_score:
            inds = valid_inds[inds]
        return inds


def nms(nms_op,
        cat_op,
        boxes: Union[Tensor, np.ndarray],
        scores: Union[Tensor, np.ndarray],
        iou_threshold: float,
        offset: int = 0,
        score_threshold: float = 0,
        max_num: int = -1,
        ) -> Tuple[Union[Tensor, np.ndarray], Union[Tensor, np.ndarray]]:
    assert isinstance(boxes, (np.ndarray, Tensor))
    assert isinstance(scores, (np.ndarray, Tensor))
    is_numpy = False
    if isinstance(boxes, np.ndarray):
        is_numpy = True
        boxes = ms.Tensor(boxes)
    if isinstance(scores, np.ndarray):
        scores = ms.Tensor(scores)
    assert boxes.shape[1] == 4
    assert boxes.shape[0] == scores.shape[0]
    assert offset in (0, 1)

    # if isinstance(boxes, Tensor):
    
    # start = time.perf_counter()
    inds = nms_op(boxes, scores, iou_threshold, offset, score_threshold, max_num)
    # topk_time = time.perf_counter()
    # print(f"nms op time: {topk_time - start}")
    # start = time.perf_counter()
    dets = cat_op((boxes[inds], scores[inds].unsqueeze(-1)))
    # topk_time = time.perf_counter()
    # print(f"cat time: {topk_time - start}")
    if is_numpy:
        dets = dets.asnumpy()
        inds = inds.asnumpy()
    return dets, inds



def ms_batched_nms(boxes: Tensor,
                scores: Tensor,
                idxs: Tensor,
                nms_cfg: Optional[Dict],
                class_agnostic: bool = False) -> Tuple[Tensor, Tensor]:
    r"""Performs non-maximum suppression in a batched fashion.
    """
    # skip nms when nms_cfg is None
    if nms_cfg is None:
        scores, inds = scores.sort(descending=True)
        boxes = boxes[inds]
        return ms.ops.cat([boxes, scores[:, None]], -1), inds

    nms_cfg_ = nms_cfg.copy()
    class_agnostic = nms_cfg_.pop('class_agnostic', class_agnostic)

    if class_agnostic:
        boxes_for_nms = boxes
    else:
        # When using rotated boxes, only apply offsets on center.
        if boxes.shape[-1] == 5:
            # Strictly, the maximum coordinates of the rotating box
            # (x,y,w,h,a) should be calculated by polygon coordinates.
            # But the conversion from rotated box to polygon will
            # slow down the speed.
            # So we use max(x,y) + max(w,h) as max coordinate
            # which is larger than polygon max coordinate
            # max(x1, y1, x2, y2,x3, y3, x4, y4)
            max_coordinate = boxes[..., :2].max() + boxes[..., 2:4].max()
            offsets = idxs.to(boxes) * (max_coordinate + ms.Tensor(1).to(boxes))
            boxes_ctr_for_nms = boxes[..., :2] + offsets[:, None]
            boxes_for_nms = ms.ops.cat([boxes_ctr_for_nms, boxes[..., 2:5]], axis=-1)
        else:
            max_coordinate = boxes.max()
            offsets = idxs.to(ms.float32) * (max_coordinate + ms.Tensor(1.).to(ms.float32))
            
            boxes_for_nms = boxes + offsets[:, None]

    nms_type = nms_cfg_.pop('type', 'nms')

    nms_op = NMSop(nms_cfg_['iou_threshold'])
    cat_op = ms.ops.Concat(1)
    # if isinstance(nms_op, str):
    #     nms_op = eval(nms_op)

    split_thr = nms_cfg_.pop('split_thr', 100000)
    # Won't split to multiple nms nodes when exporting to onnx
    if boxes_for_nms.shape[0] < split_thr:
        dets, keep = nms(nms_op,cat_op, boxes_for_nms, scores, **nms_cfg_)
        boxes = boxes[keep]

        # This assumes `dets` has arbitrary dimensions where
        # the last dimension is score.
        # Currently it supports bounding boxes [x1, y1, x2, y2, score] or
        # rotated boxes [cx, cy, w, h, angle_radian, score].

        scores = dets[:, -1]
    else:
        max_num = nms_cfg_.pop('max_num', -1)
        total_mask = scores.new_zeros(scores.shape, dtype=ms.bool_)
        # Some type of nms would reweight the score, such as SoftNMS
        scores_after_nms = scores.new_zeros(scores.shape)
        time_nms = 0
        for id in ms.ops.unique(idxs)[0]:

            mask = (idxs == id).nonzero().view(-1)
            # start = time.perf_counter()
            dets, keep = nms(nms_op,cat_op,boxes_for_nms[mask], scores[mask], **nms_cfg_)
            # nms_time = time.perf_counter()
            # time_nms += nms_time - start
            total_mask = ms.ops.tensor_scatter_elements(total_mask, mask[keep], ms.ops.full_like(mask[keep], True, dtype=ms.bool_))
            scores_after_nms = ms.ops.tensor_scatter_elements(scores_after_nms, mask[keep], dets[:, -1])

        # print(f"nms time: {time_nms}")


        keep = total_mask.nonzero().view(-1)

        scores, inds = scores_after_nms[keep].sort(descending=True)
        keep = keep[inds]
        boxes = boxes[keep]

        if max_num > 0:
            keep = keep[:max_num]
            boxes = boxes[:max_num]
            scores = scores[:max_num]

    boxes = ms.ops.cat([boxes, scores[:, None]], -1)
    return boxes, keep

