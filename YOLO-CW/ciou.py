import mindspore
from mindspore import nn, ops
import numpy as np
import mindspore
from mindspore import nn, ops
import math

class CIou(nn.Cell):
    """Calculating CIoU loss"""
    def __init__(self):
        super(CIou, self).__init__()
        self.min = ops.Minimum()
        self.max = ops.Maximum()
        self.sub = ops.Sub()
        self.add = ops.Add()
        self.mul = ops.Mul()
        self.div = ops.RealDiv()
        self.square = ops.Square()
        self.sqrt = ops.Sqrt()
        self.atan2 = ops.Atan2()
        self.eps = 1e-7
        self.pi = mindspore.Tensor(math.pi, mindspore.float32)
        self.cast = ops.Cast()

    def construct(self, boxes1, boxes2):
        """
        Args:
            boxes1: Tensor of shape (..., 4), format [xmin, ymin, xmax, ymax]
            boxes2: Tensor of shape (..., 4), format [xmin, ymin, xmax, ymax]
        Returns:
            cious: Tensor of CIoU loss values
        """
        boxes1 = self.cast(boxes1, mindspore.float32)
        boxes2 = self.cast(boxes2, mindspore.float32)

        # Widths and heights
        w1 = self.sub(boxes1[..., 2], boxes1[..., 0])
        h1 = self.sub(boxes1[..., 3], boxes1[..., 1])
        w2 = self.sub(boxes2[..., 2], boxes2[..., 0])
        h2 = self.sub(boxes2[..., 3], boxes2[..., 1])

        w1 = self.max(w1, 0.0)
        h1 = self.max(h1, 0.0)
        w2 = self.max(w2, 0.0)
        h2 = self.max(h2, 0.0)

        # Areas
        area1 = self.mul(w1, h1)
        area2 = self.mul(w2, h2)

        # Intersection
        inter_left_up = self.max(boxes1[..., :2], boxes2[..., :2])
        inter_right_down = self.min(boxes1[..., 2:], boxes2[..., 2:])
        inter_wh = self.max(self.sub(inter_right_down, inter_left_up), 0.0)
        inter_area = self.mul(inter_wh[..., 0], inter_wh[..., 1])

        # Union
        union_area = self.add(area1, area2) - inter_area + self.eps

        # IoU
        ious = self.div(inter_area, union_area)
        ious = ops.clip_by_value(ious, 0.0, 1.0)

        # Enclosing box
        enclose_left_up = self.min(boxes1[..., :2], boxes2[..., :2])
        enclose_right_down = self.max(boxes1[..., 2:], boxes2[..., 2:])
        enclose_wh = self.max(self.sub(enclose_right_down, enclose_left_up), 0.0)
        enclose_c2 = self.square(enclose_wh[..., 0]) + self.square(enclose_wh[..., 1]) + self.eps

        # Center distances
        boxes1_center = self.mul(self.add(boxes1[..., :2], boxes1[..., 2:]), 0.5)
        boxes2_center = self.mul(self.add(boxes2[..., :2], boxes2[..., 2:]), 0.5)
        center_dist = self.square(self.sub(boxes1_center[..., 0], boxes2_center[..., 0])) + \
                      self.square(self.sub(boxes1_center[..., 1], boxes2_center[..., 1]))

        # Penalty term v
        arctan1 = self.atan2(h1, w1)
        arctan2 = self.atan2(h2, w2)
        v = (4 / (self.pi ** 2)) * self.square(arctan1 - arctan2)

        # Alpha term
        S = 1 - ious
        alpha = v / (S + v + self.eps)

        # CIoU
        ciou_term = self.div(center_dist, enclose_c2)
        cious = ious - (ciou_term + alpha * v)
        cious = ops.clip_by_value(cious, -1.0, 1.0)

        return cious

# 假设有两个边界框
boxes_pred = mindspore.Tensor([[50, 50, 150, 150]], mindspore.float32)
boxes_target = mindspore.Tensor([[60, 60, 140, 140]], mindspore.float32)

# 初始化CIoU损失函数
ciou_loss = CIou()

# 计算CIoU损失
loss = ciou_loss(boxes_pred, boxes_target)
print(loss)
