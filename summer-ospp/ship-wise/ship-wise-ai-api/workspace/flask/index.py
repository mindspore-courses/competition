from io import BytesIO

import cv2
import numpy as np
from flask import Flask, request, jsonify

from workspace.flask.model.yolov8 import init, infer

app = Flask(__name__)

# 在应用开始时加载模型
user_config = {
    "config": r"H:\Workspace\DeepLearning\mindyolo-summer-ospp/workspace/configs/yolov8/yolov8s.yaml",
    "weight": r"H:\Workspace\DeepLearning\mindyolo-summer-ospp/runs/2024.09.15-22.56.30/weights/yolov8s-153_422.ckpt",
    "save_result": False,
    "device_target": "CPU",
}
args, network = init(user_config)


@app.route('/detect', methods=['POST'])
def detect():
    """
    检测图片中的物体
    输入：图片文件
    输出：{ "bbox": [[698.248,524.238,217.65,196.28]], "category_id": [18], "score": [0.82683] }
    """
    file = request.files['image']
    in_memory_file = BytesIO()
    file.save(in_memory_file)
    data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    result = infer(args, network, image)
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, port=8080)
