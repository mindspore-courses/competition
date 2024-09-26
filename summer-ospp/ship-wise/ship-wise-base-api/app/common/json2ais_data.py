# -*- coding: utf-8 -*-
# @Time    : 2024-08-25 20:46
# @Author  : Jiang Liu
import json
from linecache import cache


def json2ais_data(json_filepath):
    """
    json 格式为
    {
  "type": 1,
  "data": {
    "rows": [
      {
        "LAT": "5.414995",
        "LON": "119.7595",
        "SPEED": "131",
        "COURSE": "16",
        "HEADING": null,
        "ELAPSED": "220",
        "DESTINATION": "BR PMA>CN YNT",
        "FLAG": "HK",
        "LENGTH": "361",
        "SHIPNAME": "PACIFIC HARVEST",
        "SHIPTYPE": "7",
        "SHIP_ID": "5619078",
        "WIDTH": "65",
        "L_FORE": "309",
        "W_LEFT": "26",
        "DWT": "398544",
        "GT_SHIPTYPE": "8"
      },
    ]
    "areaShips": 562}}
    ais 数据格式为
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
    with open(json_filepath, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
        rows = json_data['data']['rows']
        ais_data = []
        for row in rows:
            try:
                ais_data.append([
                    0,  # 时间戳 (这里用0作为默认值)
                    row.get('SHIP_ID') if row.get('SHIP_ID') is not None else '0000000',
                    # MMSI (如果没有提供，使用默认值 '0000000')
                    row.get('SHIPNAME') if row.get('SHIPNAME') is not None else 'UNKNOWN',  # 船名 (默认值 'UNKNOWN')
                    row.get('FLAG') if row.get('FLAG') is not None else 'UNKNOWN',  # 呼号 (默认值 'UNKNOWN')
                    row.get('SHIP_ID') if row.get('SHIP_ID') is not None else '0000000',  # IMO (使用SHIP_ID作为IMO，如果没有提供)
                    int(row.get('SHIPTYPE')) if row.get('SHIPTYPE') is not None else 0,  # 船舶类型 (默认值 '0')
                    int(row.get('LENGTH')) if row.get('LENGTH') is not None else 0,  # 船长 (默认值 '0')
                    int(row.get('WIDTH')) if row.get('WIDTH') is not None else 0,  # 船宽 (默认值 '0')
                    int(row.get('L_FORE')) if row.get('L_FORE') is not None else 0,  # 船首向 (默认值 '0')
                    int(row.get('COURSE')) if row.get('COURSE') is not None else 0,  # 航迹向 (默认值 '0')
                    int(row.get('SPEED')) if row.get('SPEED') is not None else 0,  # 航速 (默认值 '0')
                    float(row.get('LAT')) if row.get('LAT') is not None else 0.0,  # 纬度 (默认值 '0')
                    float(row.get('LON')) if row.get('LON') is not None else 0.0,  # 经度 (默认值 '0')
                    row.get('DESTINATION') if row.get('DESTINATION') is not None else 'UNKNOWN',  # 目的地 (默认值 'UNKNOWN')
                    row.get('FLAG') if row.get('FLAG') is not None else 'UNKNOWN',  # 国家 (使用FLAG字段作为国家，默认值 'UNKNOWN')
                ])
            except Exception as e:
                print(row)
                print(f'Error: {e}')
        return ais_data


def load_ais_data(json_filepath):
    with open(json_filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def dump_data():
    json_filepaths = [
        '../../data/ais-history/z6X25Y13-202408252040.json',
        '../../data/ais-history/z6X25Y14-202408252040.json',
        '../../data/ais-history/z6X25Y15-202408252040.json',
        '../../data/ais-history/z6X26Y13-202408252040.json',
        '../../data/ais-history/z6X26Y14-202408252040.json',
        '../../data/ais-history/z6X26Y15-202408252040.json',
    ]
    ais_data = []
    for json_filepath in json_filepaths:
        ais_data.extend(json2ais_data(json_filepath))
    with open('../../data/ais-history/z6X25-26Y13-15-202408252040.json', 'w', encoding='utf-8') as f:
        json.dump(ais_data, f, ensure_ascii=False)


if __name__ == '__main__':
    dump_data()
