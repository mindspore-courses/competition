# -*- coding: utf-8 -*-
# @Time    : 2024-08-26 16:13
# @Author  : Jiang Liu
from datetime import datetime

from flask import Blueprint, jsonify, request
from flask_cors import CORS

from app.common.json2ais_data import load_ais_data
from app.common.response_wrapper import ResponseWrapper

blueprint = Blueprint('visualization', __name__)
CORS(blueprint)


@blueprint.route('/points/getVesselNum', methods=['GET'])
def get_vessel_num():
    example_ais_data_filepath = 'data/ais-history/z6X25-26Y13-15-202408252040.json'
    ais_data = load_ais_data(example_ais_data_filepath)
    res = {
        "curNum": len(ais_data),
        "stopNum": sum(1 for row in ais_data if row[10] == 0),
    }
    return jsonify(ResponseWrapper.success().set_data(res).to_dict())


@blueprint.route('/points/getSpeed', methods=['GET'])
def get_speed():
    """
    [
      [<航速>, <船舶数量>],
      ...
    ]
    :return:
    """
    example_ais_data_filepath = 'data/ais-history/z6X25-26Y13-15-202408252040.json'
    ais_data = load_ais_data(example_ais_data_filepath)
    res = {}
    for row in ais_data:
        speed = row[10]
        if speed not in res:
            res[speed] = 0
        res[speed] += 1
    res = sorted([[k, v] for k, v in res.items()])
    return jsonify(ResponseWrapper.success().set_data(res).to_dict())


@blueprint.route('/points/getVesselType', methods=['GET'])
def get_vessel_type():
    """
    [
      [<船舶类型>, <船舶数量>],
      ...
    ]
    :return:
    """
    example_ais_data_filepath = 'data/ais-history/z6X25-26Y13-15-202408252040.json'
    ais_data = load_ais_data(example_ais_data_filepath)
    res = {}
    for row in ais_data:
        vessel_type = row[5]
        if vessel_type not in res:
            res[vessel_type] = 0
        res[vessel_type] += 1
    res = sorted([[k, v] for k, v in res.items()])
    return jsonify(ResponseWrapper.success().set_data(res).to_dict())


@blueprint.route('/points/getCountryNum', methods=['GET'])
def get_country_num():
    """
    [
      [<国家名称>, <船舶数量>],
      ...
    ]
    :return:
    """
    example_ais_data_filepath = 'data/ais-history/z6X25-26Y13-15-202408252040.json'
    ais_data = load_ais_data(example_ais_data_filepath)
    res = {}
    for row in ais_data:
        country = row[14]
        if country not in res:
            res[country] = 0
        res[country] += 1
    res = sorted([[k, v] for k, v in res.items()])[1:11]
    return jsonify(ResponseWrapper.success().set_data(res).to_dict())


@blueprint.route('/points/getLatestPos', methods=['GET'])
def get_latest_pos():
    """
    [
      [<日期>, <MMSI>, <纬度>, <经度>, <航速>, <航向>, <船首向>],
      ...
    ]
    :return:
    """
    example_ais_data_filepath = 'data/ais-history/z6X25-26Y13-15-202408252040.json'
    ais_data = load_ais_data(example_ais_data_filepath)
    res = []
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    for row in ais_data:
        date = now
        mmsi = row[1]
        lat = row[11]
        lon = row[12]
        speed = row[10]
        course = row[9]
        l_fore = row[8]
        res.append([date, mmsi, lat, lon, speed, course, l_fore])
    return jsonify(ResponseWrapper.success().set_data(res).to_dict())
