# -*- coding: utf-8 -*-
# @Time    : 2024-08-25 19:30
# @Author  : Jiang Liu

from flask import Blueprint, jsonify, request
from flask_cors import CORS

from app.common.json2ais_data import json2ais_data
from app.common.response_wrapper import ResponseWrapper

blueprint = Blueprint('points', __name__)
CORS(blueprint)


@blueprint.route('/points/getTifInfo', methods=['GET'])
def get_tif_info():
    pic_name = request.args.get('pic_name', "")
    print(pic_name)
    return jsonify(ResponseWrapper.success().set_data({
        "tif_name": "<TIF文件名称>",
        "resolution": "<分辨率>",
        "size": "<文件大小>",
        "bounding_box": {
            "left": "<左边界>",
            "bottom": "<下边界>",
            "right": "<右边界>",
            "top": "<上边界>"
        },
        "creation_time": "<创建时间>"
    }).to_dict())


@blueprint.route('/points/searchPoint', methods=['GET'])
def search_point():
    """
    [[
        <MMSI>,
        <呼号>,
        <IMO>,
        <船舶类型>,
        <船舶长度>,
        <船舶宽度>,
        <最大吃水深度>,
        <船首向>,
        <航向>,
        <航速>,
        <纬度>,
        <经度>,
        <目的地>,
        <更新时间>,
        <船名>,
        <国家名称>
    ]],
    :return:
    """
    pic_name = request.args.get('pic_name', "")
    x = request.args.get('x', 0)
    y = request.args.get('y', 0)
    print(pic_name, x, y)
    data = [[
        123456789,  # MMSI
        123456789,  # 呼号
        "IMO",  # IMO
        8,  # 船舶类型
        100,  # 船舶长度
        30,  # 船舶宽度
        10,  # 最大吃水深度
        90,  # 船首向
        180,  # 航向
        20,  # 航速
        18.214594,  # 纬度
        109.479140,  # 经度
        "目的地",  # 目的地
        "更新时间",  # 更新时间
        "船名",  # 船名
        "国家名称"  # 国家名称
    ]]
    return jsonify(ResponseWrapper.success().set_data(data).to_dict())


@blueprint.route('/points/getPoints', methods=['GET'])
def get_points():
    """
    [
      [
        <时间戳>,
        <MMSI>,
        <船名>,
        <呼号>,
        <IMO>,
        <船舶类型>,
        <船长>,
        <船宽>,
        <船首向>,
        <船迹向>,
        <航速>,
        <纬度>,
        <经度>,
        <目的地>,
        <国家>
      ],
      ...
    ]
    :return:
    """
    left_down_lng = request.args.get('leftdown_lng', "")
    left_down_lat = request.args.get('leftdown_lat', "")
    right_up_lng = request.args.get('rightup_lng', "")
    right_up_lat = request.args.get('rightup_lat', "")
    example_json_data_filepaths = [
        'data/ais-history/z6X25Y13-202408252040.json',
        'data/ais-history/z6X25Y14-202408252040.json',
        'data/ais-history/z6X25Y15-202408252040.json',
        'data/ais-history/z6X26Y13-202408252040.json',
        'data/ais-history/z6X26Y14-202408252040.json',
        'data/ais-history/z6X26Y15-202408252040.json',
    ]
    example_ais_data = []
    for json_filepath in example_json_data_filepaths:
        example_ais_data.extend(json2ais_data(json_filepath))
    return jsonify(ResponseWrapper.success().set_data(example_ais_data).to_dict())
