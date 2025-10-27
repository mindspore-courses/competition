from __future__ import annotations  # noqa: F407

from typing import Any, Dict, List
import mindspore as ms
from mindspore import nn,ops

from utils.iou import iou3d, giou3d
from utils.bbox import get_box_corners
# from utils.data import decollate_batch
from utils.misc import interp



class mAP3D(nn.Cell):
    def __init__(self,
                 threshold: float = 0.3,
                 nelem: int = 101):
        """Mean average percision for 3D bounding boxes.

        Arguments:
            threshold: IoU threshold for bounding box matching.
            nelem: Number of elements to be used for the
                discretization of the precision recall curve.
        """
        super().__init__()

        self.threshold = threshold
        self.nelem = nelem

    def construct(self,
                inputs: Dict[str, ms.Tensor],
                targets: Dict[str, ms.Tensor]) -> Dict[str, ms.Tensor]:
        """Returns the mean average precision value.

        Arguments:
            inputs: This is a dict that contains at least these entries:
                "class": Bounding box class probabilities of shape (B, N, C)
                "center": Bounding box center coordinates of shape (B, N, 3).
                "size": Bounding box size values of shape (B, N, 3).
                "angle": Bounding box orientation values of shape (B, N, 2).

            targets: This is a dict of targets that contains at least these entries:
                "gt_class": Bounding box class probabilities of shape (B, M, C)
                "gt_center": Bounding box center coordinates of shape (B, M, 3).
                "gt_size": Bounding box size values of shape (B, M, 3).
                "gt_angle": Bounding box orientation values of shape (B, M, 2).

        Returns:
            mAP: Mean average precision value.
        """

        # Determine the number of classes
        num_classes = targets['gt_class'].shape[-1]

        label = ops.argmax(inputs['class'], dim=-1)
        gt_label = ops.argmax(targets['gt_class'], dim=-1)

        # Reconstruc angle from sin and cos part
        angle = ops.atan2(inputs['angle'][..., 0], inputs['angle'][..., 1])
        gt_angle = ops.atan2(targets['gt_angle'][..., 0], targets['gt_angle'][..., 1])

        # Initialize average precision values
        aps = ops.zeros((num_classes, ), dtype=ms.float32)
        idx_list = []
        for l in range(num_classes):
            # Get class label mask with shape (B, N) and (B, M)
            mask = (label == l)
            gt_mask = (gt_label == l)

            # Get 3d boundng box corners with shape (B, N, 8, 3) and (B, M, 8, 3)
            corners = get_box_corners(inputs['center'], inputs['size'], angle)
            gt_corners = get_box_corners(targets['gt_center'], targets['gt_size'], gt_angle)

            # Get box corners mask with shape (B, N, 8, 3) and (B, M, 8, 3)
            corners_mask = ops.tile(mask.unsqueeze(-1).unsqueeze(-1),(1, 1, 8, 3))
            gt_corners_mask = ops.tile(gt_mask.unsqueeze(-1).unsqueeze(-1),(1, 1, 8, 3))

            # Get intersection over union with shape (B, N, M)
            iou,idx = iou3d(ops.mul(corners, corners_mask), ops.mul(gt_corners, gt_corners_mask))
            idx_list.append(idx)
            # Flatten iou and masks along batch dimension (B, N, M) -> (B * N, M)
            iou = iou.flatten(0, 1)
            mask = mask.flatten(0, 1)
            gt_mask = gt_mask.flatten(0, 1)

            # Get number of ground truth elements
            npos = ops.sum(gt_mask).astype(ms.float32)

            # Sort iou and masks by confidence score
            sort_idx = ops.argsort(inputs['class'][..., l], descending=True).flatten(0, 1)
            iou = iou[sort_idx, :]
            mask = mask[sort_idx]
            # Get mask for all ious that are lower than the required threshold
            thr_mask = (iou > self.threshold)

            # Get final iou mask with shape (B * N, B * M)
            iou_mask = ops.logical_and(*ops.meshgrid(mask, gt_mask, indexing='ij'))

            # Get true positive candidates mask
            
            tp_c_mask = ops.logical_and(iou_mask, thr_mask)
            # 将布尔数组转换为float32类型，True->1.0, False->0.0
            tp_c_mask_float = ops.cast(tp_c_mask, ms.float32)
            
            # Initialize true positives and false positives
            tp = ops.zeros(iou.shape[0], dtype=ms.float32)
            fp = ops.ones(iou.shape[0], dtype=ms.float32)

            # Get true positives (使用float32类型的数组)
            tp_value, tp_idx = ops.max(tp_c_mask_float, axis=0)
            # 将tp_value转回布尔值用于索引
            tp_value_bool = ops.cast(tp_value, ms.bool_)
            
            # 转换为numpy数组
            tp = tp.asnumpy()
            fp = fp.asnumpy()
            tp_value_bool_np = tp_value_bool.asnumpy()
            tp_idx_np = tp_idx.asnumpy()
            
            # 使用布尔值进行索引，只选择值为True的索引
            if tp_value_bool_np.any():
                tp[tp_idx_np[tp_value_bool_np]] = 1
                fp[tp_idx_np[tp_value_bool_np]] = 0

            # Adjust for true negatives
            fp[~mask] = 0

            # 将NumPy数组转回MindSpore Tensor
            tp_tensor = ms.Tensor(tp, dtype=ms.float32)
            fp_tensor = ms.Tensor(fp, dtype=ms.float32)

            # Accumulate values (使用Tensor类型)
            tp = ops.cumsum(tp_tensor, axis=0)
            fp = ops.cumsum(fp_tensor, axis=0)

            # Calculate precision (avoid div by zero)
            prec = ops.zeros_like(tp)
            div_mask = (fp + tp != 0)
            prec[div_mask] = tp[div_mask] / (fp[div_mask] + tp[div_mask])

            # Calculate recall (avoid div by zero)
            if npos == 0:
                rec = ops.ones_like(tp)
            else:
                rec = tp / npos

            # Interpolate precision and recall
            rec_interp = ops.linspace(0, 1, self.nelem)
            prec = interp(rec_interp, rec, prec, right=0)
            rec = rec_interp

            # Calculate average precision
            aps[l] = ops.sum(prec * 1 / (self.nelem - 1))

        # Select contributing (present) classes only
        selection = ops.sort(ops.unique(ops.concat([label, gt_label], axis=1))[0])[0][1:]
        
        # Avoid empty selection
        if not selection.numel() or not selection.any():
            return ops.ones((), dtype=ms.float32)

        # Calculate mAP and ignore first class
        mAP = ops.mean(aps[selection])

        return mAP,idx_list


class mGIoU3D(nn.Cell):
    def __init__(self):
        """Generalized intersection over union.
        """
        super().__init__()

    def construct(self,
                inputs: Dict[str, ops.Tensor],
                targets: Dict[str, ops.Tensor]) -> Dict[str, ops.Tensor]:
        """Returns the generalized intersection over union value.

        Arguments:
            inputs: This is a dict that contains at least these entries:
                "class": Bounding box class probabilities of shape (B, N, C)
                "center": Bounding box center coordinates of shape (B, N, 3).
                "size": Bounding box size values of shape (B, N, 3).
                "angle": Bounding box orientation values of shape (B, N, 2).

            targets: This is a dict of targets that contains at least these entries:
                "gt_class": Bounding box class probabilities of shape (B, M, C)
                "gt_center": Bounding box center coordinates of shape (B, M, 3).
                "gt_size": Bounding box size values of shape (B, M, 3).
                "gt_angle": Bounding box orientation values of shape (B, M, 2).

        Returns:
            giou: Generalized intersection over union value.
        """
        # Get input shapes
        num_classes = targets['gt_class'].shape[-1]

        label = ops.argmax(inputs['class'], dim=-1)
        gt_label = ops.argmax(targets['gt_class'], dim=-1)

        # Reconstruc angle from sin and cos part
        angle = ops.atan2(inputs['angle'][..., 0], inputs['angle'][..., 1])
        gt_angle = ops.atan2(targets['gt_angle'][..., 0], targets['gt_angle'][..., 1])

        # Initialize giou values
        gious = -ops.ones((num_classes, ), dtype=ms.float32)

        for l in range(num_classes):
            # Get class label mask with shape (B, N) and (B, M)
            mask = (label == l)
            gt_mask = (gt_label == l)

            # Get 3d boundng box corners with shape (B, N, 8, 3) and (B, M, 8, 3)
            corners = get_box_corners(inputs['center'], inputs['size'], angle)
            gt_corners = get_box_corners(targets['gt_center'], targets['gt_size'], gt_angle)

            # Get box corners mask with shape (B, N, 8, 3) and (B, M, 8, 3)
            corners_mask = mask.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 8, 3)
            gt_corners_mask = gt_mask.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 8, 3)

            # Get intersection over union with shape (B, N, M)
            giou = giou3d(ops.mul(corners, corners_mask), ops.mul(gt_corners, gt_corners_mask))

            # Flatten iou and masks along batch dimension (B, N, M) -> (B * N, M)
            giou = giou.flatten(0, 1)
            mask = mask.flatten(0, 1)
            gt_mask = gt_mask.flatten(0, 1)

            # Sort iou and masks by confidence score
            sort_idx = ops.argsort(inputs['class'][..., l], descending=True).flatten(0, 1)
            giou = giou[sort_idx, :]
            mask = mask[sort_idx]

            # Get final iou mask with shape (B * N, B * M)
            giou_mask = ops.logical_and(*ops.meshgrid(mask, gt_mask, indexing='ij'))

            # Set unmatched values to -1
            giou[~giou_mask] = -1

            # Get most confident match
            match_giou, _ = ops.max(giou, dim=0)

            # Add class GIoU
            if gt_mask.sum() == 0:
                gious[l] = 1.0

            if match_giou.numel() > 0 and giou_mask.any():
                gious[l] = ops.mean(match_giou)

        # Select contributing (present) classes only
        selection = ops.sort(ops.unique(ops.concatenate([label, gt_label], dim=1)))[0][1:]

        # Avoid empty selection
        if not selection.numel() or not selection.any():
            return ops.ones((), dtype=ms.float32)

        # Calculate GIoU and ignore first class
        giou = ops.mean(gious[selection])     

        return giou


class Metric(nn.Cell):
    def __init__(self,
                 metrics: Dict[str, nn.Cell] = None,
                 reduction: str = 'mean',
                 **kwargs):
        """Metric module.

        Arguments:
            metrics: Dictionary of metric functions. Mapping a
                metric name to a metric function.
            reduction: Reduction mode for the per batch metric values.
                One of either none, sum or mean.
        """
        # Initialize base class
        super().__init__(**kwargs)

        # Check input arguments
        if reduction not in {'none', 'mean', 'sum'}:
            raise ValueError(
                    f"Invalid Value for arg 'reduction': '{self.reduction}"
                    f"\n Supported reduction modes: 'none', 'mean', 'sum'"
                )

        # Initialize instance attributes
        self.metrics = metrics if metrics is not None else {}
        self.reduction = reduction

        # Get reduction function
        if self.reduction != 'none':
            self.reduction_fn = getattr(ops, self.reduction)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> Metric:  # noqa: F821
        metrics = None
        reduction = config.get('reduction', 'mean')

        if 'metrics' in config:
            metrics = {k: _get_metric(v) for k, v in config['metrics'].items()}

        return cls(
            metrics=metrics,
            reduction=reduction
        )

    def construct(self,
                inputs: Dict[str, ms.Tensor],
                targets: List[Dict[str, ms.Tensor]]) -> Dict[str, ms.Tensor]:
        """Returns the loss given a prediction and ground truth.

        Arguments:
            inputs: Dictionary of model predictions with shape (B, N, C).
            targets: List of dictionaries with ground truth values
                with shape (B, M, C).

        Returns:
            metrics: Dictionary of metric values.
        """
        # Initialize losses
        batch_metrics = []
        idx_list = []
        
        # Decollate inputs
        inputs: List[Dict[str, ms.Tensor]] = decollate_batch(inputs, detach=False, pad=False)
        targets: List[Dict[str, ms.Tensor]] = decollate_batch(targets, detach=False, pad=False)

        # Get loss for each item in the batch
        for input, target in zip(inputs, targets):
            # Insert dummy batch dimension
            input = {k: v.unsqueeze(0) for k, v in input.items()}
            target = {k: v.unsqueeze(0) for k, v in target.items()}

            # Get metric values
            metrics = {name: metric(input, target)[0] for name, metric in self.metrics.items()}
            idx = [metric(input, target)[1] for _, metric in self.metrics.items()]

            # Add metrics to the batch
            batch_metrics.append(metrics)
            idx_list.append(idx)
        
        # Catch no metric configuration
        if not self.metrics:
            return ops.ones(1, dtype=ms.float32)

        # Collate metrics (revert decollating)
        # for batch_metric in batch_metrics:    
        # batch_metrics: Dict[str, ms.Tensor] = default_collate(batch_metrics)
        map_tensor = ops.stack([batch_metric['mAP'] for batch_metric in batch_metrics],axis=0)
        batch_metrics={'mAP': map_tensor}
        # Reduce batch metrics
        if self.reduction != 'none':
            batch_metrics = {k: self.reduction_fn(v) for k, v in batch_metrics.items()}

        return batch_metrics,idx_list


def _get_metric(name: str) :
    """Returns a pytorch or custom loss function given its name.

    Attributes:
        name: Name of the loss function (class).

    Returns:
        Instance of a loss function.
    """
    try:
        return getattr(nn, name)()
    except AttributeError:
        return globals()[name]()
    except Exception as e:
        raise e

# 实现decollate_batch函数用于拆分输入的批量数据
def decollate_batch(inputs: Dict[str, ms.Tensor], detach: bool = False, pad: bool = False) -> List[Dict[str, ms.Tensor]]:
    """
    将批量数据拆分为单个样本的列表
    
    Args:
        inputs: 字典类型的批量数据，每个值是形状为(B, ...)的MindSpore张量
        detach: 是否将张量从计算图中分离
        pad: 是否对不同长度的序列进行填充（此处简化实现）
        
    Returns:
        拆分后的单个样本列表，每个元素是包含单个样本数据的字典
    """
    if not inputs:
        return []
    
    # 获取批量大小（从第一个张量的0维获取）
    batch_size = next(iter(inputs.values())).shape[0]
    
    # 初始化结果列表
    decollated = [{} for _ in range(batch_size)]
    
    # 对每个键对应的张量进行拆分
    for key, tensor in inputs.items():
        # 检查张量的批量维度是否一致
        if tensor.shape[0] != batch_size:
            raise ValueError(f"张量 {key} 的批量大小与其他张量不一致")
        
        # 沿着0维拆分张量
        split_tensors = ms.ops.split(tensor, axis=0, split_size_or_sections=[1]*batch_size)
        
        # 为每个样本分配对应的数据
        for i in range(batch_size):
            # 移除单例维度并根据需要分离张量
            sample_tensor = ms.ops.squeeze(split_tensors[i], axis=0)
            if detach:
                sample_tensor = sample_tensor.detach()
            decollated[i][key] = sample_tensor
    
    return decollated

def build_metric(*args, **kwargs):
    return Metric.from_config(*args, **kwargs)