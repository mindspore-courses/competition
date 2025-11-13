import time
import cv2
import numpy as np
import torch
import torchvision

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True):
    """
    将输入图像调整为指定大小，同时保持长宽比，并用填充颜色补齐
    参数:
        img: 输入图像
        new_shape: 目标大小 (宽, 高)，默认 (640,640)
        color: 填充颜色 (BGR)
        auto: 是否自动调整到 64 的倍数
        scaleFill: 是否强制拉伸填充
        scaleup: 是否允许图像放大
    返回:
        img: 调整后图像
        ratio: 缩放比例 (w, h)
        (dw, dh): 宽高方向的填充量
    """
    shape = img.shape[:2]  # 当前尺寸 [高, 宽]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # 缩放比例 (新尺寸 / 旧尺寸)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # 只缩小不放大
        r = min(r, 1.0)

    # 计算填充
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # 缩放后的宽高
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # 剩余需要填充的宽高
    if auto:  # 自动调整到 64 的倍数
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)
    elif scaleFill:  # 强制拉伸
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2  # 左右平分
    dh /= 2  # 上下平分

    if shape[::-1] != new_unpad:  # 缩放
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    # 边缘填充
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)


def xyxy2xywh(x):
    """
    将 [x1, y1, x2, y2] 格式的框 转换为 [x, y, w, h]
    其中:
        (x1,y1) 左上角, (x2,y2) 右下角
        (x,y) 框中心坐标, (w,h) 框的宽高
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # 中心 x
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # 中心 y
    y[:, 2] = x[:, 2] - x[:, 0]  # 宽
    y[:, 3] = x[:, 3] - x[:, 1]  # 高
    return y


def non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nm=0,  # mask 的数量
):
    """
    非极大值抑制 (NMS)，去除多余的重叠检测框
    输入:
        prediction: [batch, num_boxes, 5+num_classes] 预测结果
        conf_thres: 置信度阈值
        iou_thres: IoU 阈值
        classes: 仅保留特定类别
        agnostic: 是否类别无关
        multi_label: 是否允许多标签
        max_det: 最大检测数量
        nm: mask 数量 (YOLO-seg 用)
    输出:
        每张图像的检测框列表 (n, 6)，格式 [x1, y1, x2, y2, conf, cls]
    """

    if isinstance(prediction, (list, tuple)):  # 如果模型输出是 (inference, loss)
        prediction = prediction[0]

    device = prediction.device
    mps = 'mps' in device.type  # 苹果 M1/M2 上的 MPS
    if mps:  # 不完全支持 NMS, 转到 CPU
        prediction = prediction.cpu()
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - nm - 5  # 类别数
    xc = prediction[..., 4] > conf_thres  # 置信度筛选

    # 参数检查
    assert 0 <= conf_thres <= 1, f'无效置信度阈值 {conf_thres}'
    assert 0 <= iou_thres <= 1, f'无效IoU阈值 {iou_thres}'

    max_wh = 7680  # 最大框宽高
    max_nms = 30000  # NMS最大处理框数
    time_limit = 0.5 + 0.05 * bs  # 超时退出
    redundant = True  # 是否保留冗余框
    multi_label &= nc > 1
    merge = False  # 是否使用合并NMS

    t = time.time()
    mi = 5 + nc  # mask 起始索引
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # 遍历每张图像
        x = x[xc[xi]]  # 过滤低置信度框

        # 拼接标注框 (自动标注时用)
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls one-hot
            x = torch.cat((x, v), 0)

        if not x.shape[0]:  # 没有框
            continue

        # 更新置信度 conf = obj_conf * cls_conf
        x[:, 5:] *= x[:, 4:5]

        # 坐标变换 [cx,cy,w,h] -> [x1,y1,x2,y2]
        box = xywh2xyxy(x[:, :4])
        mask = x[:, mi:]

        # 构造最终输出 (xyxy, conf, cls, mask)
        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
        else:
            conf, j = x[:, 5:mi].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # 类别过滤
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        n = x.shape[0]  # 框数
        if not n:
            continue
        elif n > max_nms:  # 框太多，只保留置信度最高的 max_nms 个
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]
        else:
            x = x[x[:, 4].argsort(descending=True)]

        # NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # 按类别分组偏移
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # torchvision 提供的NMS
        if i.shape[0] > max_det:
            i = i[:max_det]

        # 可选: 合并NMS (加权平均)
        if merge and (1 < n < 3E3):
            iou = box_iou(boxes[i], boxes) > iou_thres
            weights = iou * scores[None]
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)
            if redundant:
                i = i[iou.sum(1) > 1]

        output[xi] = x[i]
        if mps:
            output[xi] = output[xi].to(device)
        if (time.time() - t) > time_limit:  # 超时退出
            break

    return output

def xywh2xyxy(x):
    """ [cx,cy,w,h] -> [x1,y1,x2,y2] """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """
    将推理时的坐标映射回原始图像尺寸
    """
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, shape):
    """将边界框裁剪到图像范围内"""
    if isinstance(boxes, torch.Tensor):
        boxes[:, 0].clamp_(0, shape[1])
        boxes[:, 1].clamp_(0, shape[0])
        boxes[:, 2].clamp_(0, shape[1])
        boxes[:, 3].clamp_(0, shape[0])
    else:
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])


def nms(box_out, conf_thres=0.4, iou_thres=0.5):
    """
    简化版NMS调用，增加了 multi_label 兼容
    """
    try:
        boxout = non_max_suppression(box_out, conf_thres=conf_thres, iou_thres=iou_thres, multi_label=True)
    except:
        boxout = non_max_suppression(box_out, conf_thres=conf_thres, iou_thres=iou_thres)
    return boxout
