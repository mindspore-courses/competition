# -*- coding: utf-8 -*-
# @Time    : 2024-08-25 18:10
# @Author  : Jiang Liu
import os

from flask import Blueprint, jsonify, request
from flask_cors import CORS

from app.common.response_wrapper import ResponseWrapper

blueprint = Blueprint('detection', __name__)
CORS(blueprint)


@blueprint.route('/detection/getDetections', methods=['GET'])
def get_detections():
    stime = request.args.get('stime', 0)
    etime = request.args.get('etime', 0)
    pic_name = request.args.get('pic_name', "")
    print(stime, etime)
    images = os.listdir("data/all-detection-images")
    images = [[
        image,
        # 以下是一些假数据
        109.479140,  # 经度
        18.214594,  # 纬度
        "",  # 子图路径
        0.9,  # 置信度
        10,  # 船只大小
        20,  # 边界框上边
        30,  # 边界框左边
        40,  # 边界框下边
        50,  # 边界框右边
    ] for image in images]
    return jsonify(ResponseWrapper.success().set_data(images).to_dict())


@blueprint.route('/detection/getSizes', methods=['GET'])
def get_sizes():
    sizes = [10, 20, 30, 40, 50]
    return jsonify(ResponseWrapper.success().set_data(sizes).to_dict())
