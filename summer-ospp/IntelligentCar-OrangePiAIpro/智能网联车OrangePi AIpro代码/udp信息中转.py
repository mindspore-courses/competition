import socket
import json
import threading
import time

# 服务端配置
LOCAL_ADDR = '0.0.0.0'  # 监听所有 IPv4 地址
LOCAL_PORT = 7788  # 服务端端口
BUFFER_SIZE = 1024

# 客户端配置
REMOTE_IP = '192.168.100.5'  # C-based UDP server IP
REMOTE_PORT = 8888       # C-based UDP server port
CLIENT_BUFFER_SIZE = 1024

# 创建 UDP 套接字 (IPv4) for server
server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

try:
    server_socket.bind((LOCAL_ADDR, LOCAL_PORT))
    print(f"UDP 服务端已启动，监听 {LOCAL_ADDR}:{LOCAL_PORT}")
except Exception as e:
    print(f"绑定套接字错误: {e}")
    exit(1)

# 模拟小车状态
car_state = {
    "carStatus": "off",
    "autoMode": 0,
    "carSpeed": "low"
}

def process_command(data):
    """处理接收到的命令并返回响应"""
    try:
        # 解析客户端发送的 JSON 数据
        command = json.loads(data.decode('utf-8'))
        response = {"status": "success", "message": ""}

        # 处理小车状态命令
        if "carStatus" in command:
            car_state["carStatus"] = command["carStatus"]
            response["message"] = f"小车状态更新为: {car_state['carStatus']}"

        # 处理自动模式命令
        elif "autoMode" in command:
            car_state["autoMode"] = command["autoMode"]
            response["message"] = f"自动模式: {'开启' if car_state['autoMode'] else '关闭'}"

        # 处理速度命令
        elif "carSpeed" in command:
            car_state["carSpeed"] = command["carSpeed"]
            response["message"] = f"小车速度设置为: {car_state['carSpeed']}"

        else:
            response = {"status": "error", "message": "未知命令"}

        # 打印当前小车状态
        print(f"当前小车状态: {car_state}")
        return json.dumps(response).encode('utf-8'), command

    except json.JSONDecodeError:
        return json.dumps({"status": "error", "message": "JSON 解析错误"}).encode('utf-8'), None
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)}).encode('utf-8'), None

def udp_server():
    """UDP 服务端主函数"""
    # 创建用于转发的 IPv4 UDP 客户端套接字
    forward_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    while True:
        try:
            # 接收客户端数据
            data, client_addr = server_socket.recvfrom(BUFFER_SIZE)
            print(f"收到来自 {client_addr} 的数据: {data.decode('utf-8')}")

            # 处理命令并生成响应
            response, command = process_command(data)

            # 发送响应回客户端
            server_socket.sendto(response, client_addr)
            print(f"发送响应到 {client_addr}: {response.decode('utf-8')}")

            # 如果命令有效，将其转发给 IPv4 UDP 服务端
            if command:
                cmd_json = json.dumps(command).encode('utf-8')
                sent = forward_socket.sendto(cmd_json, (REMOTE_IP, REMOTE_PORT))
                print(f"转发命令到 {REMOTE_IP}:{REMOTE_PORT}: {cmd_json.decode('utf-8')}")
                print(f"已发送 {sent} 字节到小车 {REMOTE_IP}:{REMOTE_PORT}")
        except KeyboardInterrupt:
            print("\n服务端关闭")
            break
        except Exception as e:
            print(f"服务端处理错误: {e}")

    # 关闭套接字
    forward_socket.close()
    server_socket.close()

def main():
    # 只启动服务端线程
    server_thread = threading.Thread(target=udp_server, daemon=True)
    server_thread.start()

    # 主线程保持运行
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n程序退出")

if __name__ == "__main__":
    main()
