import cv2
import socket
import json
from vision_model import find_camera_index, init_model, detect_lane_lines, infer_frame_with_vis, img2bytes
from queue import Queue
from collections import Counter

# ========================
# 线程安全队列（用于跨线程共享数据）
# ========================
lane_data_queue = Queue()           # 存放车道线检测指令 (F/L/R)
detected_objects_queue = Queue()    # 存放目标检测结果 (dict 计数)

# ========================
# 模型推理参数配置
# ========================
cfg = {
    'conf_thres': 0.6,        # 检测框的置信度阈值
    'iou_thres': 0.6,         # NMS 的 IOU 阈值
    'input_shape': [640, 640] # 模型输入尺寸
}

# ========================
# UDP 通信配置
# ========================
UDP_IP = "172.20.10.12"  # 接收端 IP（目标设备）
# UDP_IP = "192.168.3.26"  # 可切换为备用 IP
UDP_PORT = 5006          # 接收端端口
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# ========================
# 视频帧生成器函数 (供 Flask/流媒体使用)
# ========================
def gen_frames():
    # 获取可用的摄像头索引
    camera_index = find_camera_index()
    cap = cv2.VideoCapture(camera_index)

    # 加载模型和标签文件
    model, labels_dict = init_model()

    while True:
        # 读取摄像头画面
        ret, frame = cap.read()
        if not ret:
            break

        # 将帧缩放到模型输入大小
        frame = cv2.resize(frame, (640, 640))

        # ------------------------
        # 车道线检测
        # ------------------------
        lane_image, lanes = detect_lane_lines(frame)  # 返回叠加车道线图像和左右车道线参数
        offset = None
        command = "F"   # 默认前进

        # 根据车道线位置计算车辆偏移量和控制指令
        if lanes[0] is not None and lanes[1] is not None:  # 同时检测到左右车道线
            lane_center = (lanes[0][0] + lanes[0][2] + lanes[1][0] + lanes[1][2]) // 4
            car_center = frame.shape[1] // 2
            offset = lane_center - car_center
            if offset > 30:
                command = "R"   # 偏右，向右修正
            elif offset < -30:
                command = "L"   # 偏左，向左修正
        elif lanes[0] is None and lanes[1] is not None:   # 只检测到右车道
            command = "L"
        elif lanes[0] is not None and lanes[1] is None:   # 只检测到左车道
            command = "R"

        # 更新车道线检测队列（保证只有最新数据）
        if not lane_data_queue.empty():
            lane_data_queue.get()
        lane_data_queue.put(command)

        # ------------------------
        # 目标检测
        # ------------------------
        image_pred, detected_objects = infer_frame_with_vis(frame, model, labels_dict, cfg)
        obj_counter = Counter(detected_objects)  # 统计每类目标数量
        detected_objects_queue.put(obj_counter)

        # ------------------------
        # 通过 UDP 发送 JSON 数据
        # ------------------------
        udp_payload = {
            "lane_offset": offset if offset is not None else None,  # 车辆偏移量
            "command": command,                                    # 转向指令
            "detected_objects": dict(obj_counter)                  # 检测到的物体统计
        }
        udp_socket.sendto(json.dumps(udp_payload).encode(), (UDP_IP, UDP_PORT))

        # ------------------------
        # 将车道线与检测框叠加到一张画面
        # ------------------------
        combined_image = cv2.addWeighted(image_pred, 0.7, lane_image, 0.3, 0)

        # 将图像编码为字节流 (JPEG) 用于视频流输出
        frame_bytes = img2bytes(combined_image)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
