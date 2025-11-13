import cv2
import numpy as np
import torch
from ais_bench.infer.interface import InferSession
from det_utils import letterbox, scale_coords, nms
from collections import deque
from PIL import Image, ImageDraw, ImageFont

# =============================
# 全局变量
# =============================

# 车道线平滑用的历史缓存（deque 可自动丢弃旧数据）
left_lane_history  = deque(maxlen=5)   # 左车道历史
right_lane_history = deque(maxlen=5)   # 右车道历史

# 中文字体路径（用于在图像上绘制中文标签）
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font = ImageFont.truetype(FONT_PATH, 20, encoding="utf-8")

# =============================
# 图像预处理
# =============================
def preprocess_image(image, cfg, bgr2rgb=True):
    """
    将原始图像处理为模型输入：
      1. letterbox → 调整比例并补边
      2. 转换为 CHW 格式
      3. 转换为 float32
    """
    img, ratio, pad = letterbox(image, new_shape=cfg['input_shape'])
    if bgr2rgb:  # OpenCV 默认是 BGR，这里转成 RGB
        img = img[:, :, ::-1]
    img = img.transpose(2, 0, 1)  # HWC → CHW
    img = np.ascontiguousarray(img, dtype=np.float32)
    return img, ratio, pad

# =============================
# 画检测框
# =============================
def draw_bbox(det, img0, conf_thres, names, color=(0,255,0), wt=2):
    """
    在图像上绘制检测框和中文标签
    det: numpy (N,6) = [x1, y1, x2, y2, conf, cls]
    names: 类别字典 {id: "类别名"}
    """
    img_pil = Image.fromarray(cv2.cvtColor(img0, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    detected = []  # 记录检测到的类别
    for x1,y1,x2,y2,conf,cls in det:
        if conf < conf_thres:  # 过滤低置信度目标
            continue
        cls_id = int(cls)
        label = f"{names[cls_id]} {conf:.2f}"

        # 画矩形框
        draw.rectangle([(x1,y1),(x2,y2)], outline=tuple(color), width=wt)
        # 绘制中文标签
        draw.text((x1, max(y1-25,0)), label, font=font, fill=tuple(color))
        detected.append(names[cls_id])

    # 转回 OpenCV 格式
    img_out = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img_out, detected

# =============================
# 加载类别标签
# =============================
def get_labels_from_txt(path):
    """从 txt 文件中加载类别名，每行一个类别"""
    labels = {}
    with open(path, 'r', encoding='gbk', errors='ignore') as f:
        for i, line in enumerate(f):
            labels[i] = line.strip()
    return labels

# =============================
# 单帧推理 + 可视化
# =============================
def infer_frame_with_vis(image, model, labels, cfg):
    """
    核心推理流程：
      1. 预处理图像
      2. 模型推理
      3. NMS 过滤
      4. 坐标映射回原图
      5. 绘制检测框
    返回：带框图像, 检测类别列表
    """
    # 1) 预处理
    img, ratio, pad = preprocess_image(image, cfg)

    # 2) 推理输入 (保持 float32 格式)
    tensor = img.astype(np.float32)
    out = model.infer([tensor])[0]

    # 3) NMS 去重
    out_t = torch.tensor(out)
    dets = nms(out_t,
               conf_thres=cfg['conf_thres'],
               iou_thres=cfg['iou_thres'])
    if len(dets) == 0:
        det = np.zeros((0,6), dtype=np.float32)
    else:
        det = dets[0].numpy()

    # 4) 坐标缩放回原图尺寸
    if det.shape[0] > 0:
        scale_coords(cfg['input_shape'], det[:,:4], image.shape, ratio_pad=(ratio,pad))

    # 5) 绘制检测框
    vis, detected = draw_bbox(det, image.copy(), cfg['conf_thres'], labels)
    return vis, detected

# =============================
# 工具函数
# =============================
def img2bytes(image):
    """OpenCV 图像编码为 JPEG → bytes"""
    return bytes(cv2.imencode('.jpg', image)[1])

def find_camera_index(max_checks=5):
    """遍历查找可用摄像头索引"""
    for i in range(max_checks):
        cap = cv2.VideoCapture(i)
        if cap.read()[0]:
            cap.release()
            return i
        cap.release()
    return None

def init_model(om_path='yolo.om', label_path='coco_names.txt'):
    """初始化推理模型 & 加载类别标签"""
    sess   = InferSession(0, om_path)
    labels = get_labels_from_txt(label_path)
    return sess, labels

# =============================
# 车道线检测
# =============================
def detect_lane_lines(image):
    """
    基于 Canny + 霍夫直线变换检测车道线
    1. 灰度化 + 高斯模糊
    2. Canny 边缘检测
    3. 提取 ROI (多边形掩码)
    4. HoughLinesP 提取直线
    5. 根据斜率区分左右车道
    6. 使用历史缓存平滑
    """
    global left_lane_history, right_lane_history
    gray   = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur   = cv2.GaussianBlur(gray,(5,5),0)
    edges  = cv2.Canny(blur,30,150)

    H, W   = edges.shape
    mask   = np.zeros_like(edges)
    poly   = np.array([[(0,H),(W,H),(W//2+100,H//2),(W//2-100,H//2)]],np.int32)
    cv2.fillPoly(mask, poly, 255)
    cropped= edges & mask

    # 霍夫直线变换
    lines = cv2.HoughLinesP(cropped,1,np.pi/180,50,minLineLength=40,maxLineGap=150)
    left, right = [], []
    if lines is not None:
        for x1,y1,x2,y2 in lines[:,0]:
            slope = (y2-y1)/(x2-x1) if x2!=x1 else 1e6
            if slope < -0.5:
                left.append((x1,y1,x2,y2))   # 左车道
            elif slope > 0.5:
                right.append((x1,y1,x2,y2))  # 右车道

    # 平滑函数：取历史均值
    def smooth(fits,hist):
        if fits:
            hist.append(np.mean(fits,axis=0))
        return np.mean(hist,axis=0) if hist else None

    l = smooth(left,  left_lane_history)
    r = smooth(right, right_lane_history)

    # 绘制结果层
    layer = np.zeros_like(image)
    for lane in (l,r):
        if lane is not None:
            x1,y1,x2,y2 = lane.astype(int)
            cv2.line(layer,(x1,y1),(x2,y2),(0,0,255),8)
    return layer, (l,r)
