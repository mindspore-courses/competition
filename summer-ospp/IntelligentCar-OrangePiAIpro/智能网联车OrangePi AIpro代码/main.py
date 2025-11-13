from flask import Flask, request, jsonify, Response, render_template
from car_utils import gen_frames, detected_objects_queue, lane_data_queue
import threading
import time
from werkzeug.serving import run_simple

app = Flask(__name__)

# ========================
# 指令发送控制参数
# ========================
last_command_time = time.time()  # 上次车道保持指令发送时间
command_interval = 2.0           # 车道保持指令最小间隔 (秒)

last_send_time = time.time()     # 上次目标检测指令发送时间
send_interval = 3.0              # 目标检测指令最小间隔 (秒)

# ========================
# 线程控制标志位 (Event 用于线程安全地启停)
# ========================
lane_keeping_active = threading.Event()        # 车道保持线程开关
detection_monitor_active = threading.Event()   # 目标检测线程开关

# ========================
# 后台线程：检测目标并发送控制指令
# ========================
def monitor_detected_objects():
    """后台线程：监听 detected_objects_queue，根据检测到的目标决定是否发指令"""
    global last_send_time
    while detection_monitor_active.is_set():  # 当检测线程被激活时循环执行
        if not detected_objects_queue.empty():  # 如果队列中有新检测结果
            detected_objects = detected_objects_queue.get()
            current_time = time.time()

            # 根据检测到的目标发送指令，并控制最小时间间隔
            if detected_objects == "left" and current_time - last_send_time >= send_interval:
                send_command("L")
                last_send_time = current_time
                print("目标检测发送指令：L")
            elif detected_objects == "right" and current_time - last_send_time >= send_interval:
                send_command("R")
                last_send_time = current_time
                print("目标检测发送指令：R")

        time.sleep(0.5)  # 每 0.5 秒检查一次队列

# ========================
# 后台线程：车道保持逻辑
# ========================
def process_lane_keeping():
    """后台线程：监听 lane_data_queue，根据车道线结果发送转向指令"""
    global last_command_time
    while lane_keeping_active.is_set():  # 当车道保持线程被激活时循环执行
        command = None
        if not lane_data_queue.empty():
            command = lane_data_queue.get()  # 从队列中取出最新车道保持命令

        current_time = time.time()
        # 若有有效命令，且与上次发送的时间间隔超过 command_interval
        if command and (current_time - last_command_time >= command_interval):
            send_command(command)  # 执行命令（函数需在 ctrlCar.py 定义）
            print("车道保持发送指令：" + command)
            last_command_time = current_time

        time.sleep(0.1)  # 每 0.1 秒检查一次队列，提高响应速度

# ========================
# 自动模式：启动两个后台线程
# ========================
def automatic_mode():
    # 开启车道保持线程
    lane_keeping_active.set()
    lane_keeping_thread = threading.Thread(target=process_lane_keeping, daemon=True)
    lane_keeping_thread.start()

    # 开启目标检测线程
    detection_monitor_active.set()
    monitor_thread = threading.Thread(target=monitor_detected_objects, daemon=True)
    monitor_thread.start()

# ========================
# Flask 路由
# ========================
@app.route('/video')
def video():
    """视频流接口：实时返回 gen_frames() 生成的图像流"""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ========================
# 启动 Flask 服务
# ========================
if __name__ == '__main__':
    # 绑定到所有 IP (0.0.0.0)，端口 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
    # app.run(host='::', port=5000, debug=True)  # IPv6 支持
