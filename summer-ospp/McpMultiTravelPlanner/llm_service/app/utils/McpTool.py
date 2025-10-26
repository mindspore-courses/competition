import codecs
import json
import re
from typing import List, Dict
import pandas as pd
import requests
from mcp.server.fastmcp import FastMCP
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime, timedelta
from app import model, tokenizer, torch_device
import time
from .TourismDataSimplifier import TourismDataSimplifier

# 初始化MCP服务器
mcp = FastMCP("travel")

# 常量定义
AK = 'XXXX' # 替换为自己的百度地图AK
BAIDU_WEATHER_URL_BASE = "https://api.map.baidu.com/weather/v1/"
BAIDU_ROUTE_URL_BASE = "https://api.map.baidu.com/directionlite/v1/"
BAIDU_LOCATION_CODE_BASE = "https://api.map.baidu.com/geocoding/v3/"
BAIDU_LOCATION_SEARSH_BASE = "https://api.map.baidu.com/place/v2/search"
BAIDU_LOCATION_DETIAL_BASE = "https://api.map.baidu.com/place/v2/detail"

LITTEL_RED_URL_BASE = "https://api.coze.cn/v1/workflow/stream_run"
HOTEL_URL_BASE = "https://api.coze.cn/v1/workflow/stream_run"
GAODE_KEY = "xxxx" # 替换为自己的高德地图key
GAODE_ATTRACTIONS_URL_BASE = "https://restapi.amap.com/v3/place/text"

# 读取地区code
region_ids = pd.read_excel("./app/utils/weather_district_id.xlsx")


# 工具类定义
class TravelTools:
    @staticmethod
    def get_attractions(city: str) -> str:
        """获取某个城市的景点列表（biz_ext:cost 代表门票价格或者人均消费）。city: 城市名"""
        params = {
            "key": GAODE_KEY,
            "keywords": "景点",
            "city": city,
        }
        respones = requests.get(url=GAODE_ATTRACTIONS_URL_BASE, params=params)
        data = respones.json()
        if not data:
            print(respones.text)
            return "景区信息获取失败， 请稍后再试。"
        return json.dumps(data)

    @staticmethod
    def get_hotels(location: str) -> str:
        """获取指定详细地址周围的酒店。location: 详细地址; """
        # 首先根据地点名称获取经纬度
        params = {
            "address":location,
            "ak": AK,
            "output": "json",
        }
        respones = requests.get(url=BAIDU_LOCATION_CODE_BASE, params=params)
        data = respones.json()
        if not data  or data['status'] != 0:
            print(respones.text)
            return "经纬度编码失败， 请完善地址结构后再试。"
        location_code = json.dumps(data)
        # 从 JSON 字符串中解析数据
        parsed_data = json.loads(location_code)
        lng = parsed_data['result']['location']['lng']
        lat = parsed_data['result']['location']['lat']
        params = {
            "query": "酒店",
            "location": f"{lat},{lng}",
            "ak": AK,
            "output": "json",
            "radius": 2000,
        }
        respones = requests.get(url=BAIDU_LOCATION_SEARSH_BASE, params=params)
        data = respones.json()
        if not data or "message" not in data or data['message'] != 'ok':
            print(respones.text)
            return "酒店搜索失败， 请稍后再试。"+respones.text
        lists =  json.dumps(data)
        # 搜索指定地区的详细信息
        # 从 JSON 字符串解析回字典
        parsed_data = json.loads(lists)
        uids = [result['uid'] for result in parsed_data['results']]
        ans = []
        for uid in uids:
            params = {
                "uid": uid,
                "scope": 2,
                "ak": AK,
                "output": "json",
            }
            respones = requests.get(url=BAIDU_LOCATION_DETIAL_BASE, params=params)
            data = respones.json()
            ans.append(data)
        return ans
    
    @staticmethod
    def get_red_blog(district: str) -> str:
        """获取小红书上相关的旅游攻略帖子作为参考。district: 地区名称"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer cztei_lgqMyGdPED1MihvHNRKJTnGeRumB6IUch0GJfCFcKr7KvIXX6oU0W165puuPATDjV",
        }
        params = {
            "workflow_id": "7525656003343040527",
            "parameters": {
                "input": f"{district}旅游攻略",
            }
        }
        respones = requests.post(url=LITTEL_RED_URL_BASE, headers=headers, data=json.dumps(params))
        s = respones.text.split('\n')
        for item in s:
            if item[:4] == 'data':
                ans = json.loads(item[5:])
                return codecs.escape_decode(json.loads(ans['content'])['output'][0]['data']['note']['note_desc'].encode('unicode_escape'))[0].decode('utf-8')
        print(respones.text)
        return "参考攻略获取失败，暂时不参考。"
    
    @staticmethod
    def get_route(origin: str, destination: str, tp: str = "driving") -> str:
        """获取两地之间的交通路线（交通方式：walking、transit、driving、bicycling）。 origin:起始地经纬度(经度纬度之间用,分隔); destination:目的地经纬度; tp:交通方式"""
        params = {
            "origin": origin,
            "destination": destination,
            "ak": AK,
        }
        respones = requests.get(url=BAIDU_ROUTE_URL_BASE+tp, params=params)
        data = respones.json()
        if not data:
            print(respones.text)
            return "路线信息获取失败， 请稍后再试。"
        return json.dumps(data)
    
    @staticmethod
    def get_all_weather(district: str) -> str:
        """获取某个地区的天气。 district: 地区名称"""
        district_id = region_ids.loc[region_ids['district'] == district, 'district_geocode'].values[0]
        params = {
            "district_id": district_id,
            "data_type": "all",
            "ak": AK,
        }
        respones = requests.get(url=BAIDU_WEATHER_URL_BASE, params=params)
        data = respones.json()
        if not data or "message" not in data or data['message'] != 'success':
            print(respones.text)
            return "天气信息获取失败， 请稍后再试。"
        return json.dumps(data)
    
    @staticmethod
    def get_location_code(locationName: str) -> str:
        """"获取某个地点的经纬度。 locationName: 地点名称"""
        params = {
            "address":locationName,
            "ak": AK,
            "output": "json",
        }
        respones = requests.get(url=BAIDU_LOCATION_CODE_BASE, params=params)
        data = respones.json()
        if not data or "message" not in data or data['message'] != 'success':
            print(respones.text)
            return "经纬度编码失败， 请完善地址结构后再试。"
        return json.dumps(data)
    
    @staticmethod
    def location_search(query: str, location: str) -> str:
        """获取给定景点附近的相关关键字地点。 query:检索关键字（如美食、银行...）;location: 地点经纬度(经度纬度之间用,分隔)"""
        params = {
            "query": query,
            "location": location,
            "ak": AK,
            "output": "json",
        }
        respones = requests.get(url=BAIDU_LOCATION_SEARSH_BASE, params=params)
        data = respones.json()
        if not data or "message" not in data or data['message'] != 'success':
            print(respones.text)
            return "地点搜索失败， 请稍后再试。"
        return json.dumps(data)
    
    @staticmethod
    def get_location_detail(uid: str, location: str) -> str:
        """获取指定地点的详细信息。uid: 利用location_search工具获得"""
        params = {
            "uid": uid,
            "scope": 2,
            "ak": AK,
            "output": "json",
        }
        respones = requests.get(url=BAIDU_LOCATION_DETIAL_BASE, params=params)
        data = respones.json()
        if not data or "message" not in data or data['message'] != 'success':
            print(respones.text)
            return "地点搜索失败， 请稍后再试。"
        return json.dumps(data)

    @staticmethod
    def get_location_distance(locations: str) -> str:
        """获得给定地址名称列表之间的两两一组之间的步行距离。
        locations: 格式为 /['地点名称', '地点名称', '地点名称', ...]/
        """
        data_list = eval(locations)
        codes = []
        for loc in data_list:
            # 获取地点的经纬度
            params = {
                "address":loc,
                "ak": AK,
                "output": "json",
            }
            respones = requests.get(url=BAIDU_LOCATION_CODE_BASE, params=params)
            data = respones.json()
            if not data  or data['status'] != 0:
                print(respones.text)
                return "经纬度编码失败， 请完善地址结构后再试。"
            location_code = json.dumps(data)
            # 从 JSON 字符串中解析数据
            parsed_data = json.loads(location_code)
            lng = parsed_data['result']['location']['lng']
            lat = parsed_data['result']['location']['lat']
            codes.append(f"{lat},{lng}")
        distances  = []
        for idx1, loc1 in enumerate(codes):
            now = []
            for idx2, loc2 in enumerate(codes):
                if loc1 == loc2:
                    now.append(0)
                elif idx1>idx2:
                    now.append(distances[idx2][idx1])
                else:
                    # 获取两地之间的步行距离
                    params = {
                        "origin": loc1,
                        "destination": loc2,
                        "ak": AK,
                    }
                    respones = requests.get(url=BAIDU_ROUTE_URL_BASE+"walking", params=params)
                    data = respones.json()
                    if not data:
                        print(respones.text)
                        return "路线信息获取失败， 请稍后再试。"
                    dist =  json.dumps(data)
                    parsed_data = json.loads(dist)
                    now.append(parsed_data['result']['routes'][0]["distance"])
                    time.sleep(0.5)
            distances.append(now)
        return distances
    @staticmethod
    def get_foods(location: str) -> str:
        """获取指定详细地址周围的美食。location: 详细地址; """
        # 首先根据地点名称获取经纬度
        params = {
            "address":location,
            "ak": AK,
            "output": "json",
        }
        respones = requests.get(url=BAIDU_LOCATION_CODE_BASE, params=params)
        data = respones.json()
        if not data  or data['status'] != 0:
            print(respones.text)
            return "经纬度编码失败， 请完善地址结构后再试。"
        location_code = json.dumps(data)
        # 从 JSON 字符串中解析数据
        parsed_data = json.loads(location_code)
        lng = parsed_data['result']['location']['lng']
        lat = parsed_data['result']['location']['lat']
        params = {
            "query": "美食",
            "location": f"{lat},{lng}",
            "ak": AK,
            "output": "json",
            "radius": 2000,
        }
        respones = requests.get(url=BAIDU_LOCATION_SEARSH_BASE, params=params)
        data = respones.json()
        if not data or "message" not in data or data['message'] != 'ok':
            print(respones.text)
            return "美食搜索失败， 请稍后再试。"+respones.text
        return json.dumps(data)
    
    @staticmethod
    def get_tow_poi_transport(lpoi: str, rpoi : str) -> list:
        # 首先根据地点名称获取经纬度
        params = {
            "address":lpoi,
            "ak": AK,
            "output": "json",
        }
        respones = requests.get(url=BAIDU_LOCATION_CODE_BASE, params=params)
        data = respones.json()
        if not data  or data['status'] != 0:
            print(respones.text)
            return "经纬度编码失败， 请完善地址结构后再试。"
        location_code = json.dumps(data)
        # 从 JSON 字符串中解析数据
        parsed_data = json.loads(location_code)
        lng = parsed_data['result']['location']['lng']
        lat = parsed_data['result']['location']['lat']
        lloc = f"{lat},{lng}"

        params = {
            "address":rpoi,
            "ak": AK,
            "output": "json",
        }
        respones = requests.get(url=BAIDU_LOCATION_CODE_BASE, params=params)
        data = respones.json()
        if not data  or data['status'] != 0:
            print(respones.text)
            return "经纬度编码失败， 请完善地址结构后再试。"
        location_code = json.dumps(data)
        # 从 JSON 字符串中解析数据
        parsed_data = json.loads(location_code)
        lng = parsed_data['result']['location']['lng']
        lat = parsed_data['result']['location']['lat']
        rloc = f"{lat},{lng}"
        print(lloc, rloc)

        # 分别计算不同的交通方式信息
        ans = []
        types = ['walking','driving', 'riding', 'transit']
        for tp in types:
            params = {
                "origin": lloc,
                "destination": rloc,
                "ak": AK,
            }
            respones = requests.get(url=BAIDU_ROUTE_URL_BASE+tp, params=params)
            print(tp, lloc, rloc)
            data = respones.json()
            if not data:
                print(respones.text)
                return "路线信息获取失败， 请稍后再试。"
            dist =  json.dumps(data)
            parsed_data = json.loads(dist)
            ans.append(parsed_data)
            time.sleep(0.5)
        return ans

# 工具描述
TOOL_DESCRIPTIONS = {
    "get_attractions": {
        "description": "获取某个城市的景点列表",
        "parameters": {
            "city": {"type": "string", "description": "城市名"}
        }
    },
    "get_hotels": {
        "description": "获取指定地区指定时间段的可选酒店",
        "parameters": {
            "district": {"type": "string", "description": "城市名"},
            "checkin": {"type": "string", "description": "入住日期（格式为yyyy-mm-dd）", "required": True},
            "checkout": {"type": "string", "description": "退房日期（格式为yyyy-mm-dd）", "required": True}
        }
    },
    # "get_red_blog": {
    #     "description": "获取小红书上相关的旅游攻略帖子作为参考",
    #     "parameters": {
    #         "district": {"type": "string", "description": "地区名称"}
    #     }
    # },
    "get_route": {
        "description": "获取两地之间的交通路线",
        "parameters": {
            "origin": {"type": "string", "description": "起始地经纬度(经度纬度之间用,分隔)"},
            "destination": {"type": "string", "description": "目的地经纬度"},
            "tp": {"type": "string", "description": "交通方式：walking、transit、driving、riding", "required": False}
        }
    },
    "get_all_weather": {
        "description": "获取某个地区的天气",
        "parameters": {
            "district": {"type": "string", "description": "地区名称"}
        }
    },
    # "get_location_code": {
    #     "description": "获取某个地点的经纬度",
    #     "parameters": {
    #         "locationName": {"type": "string", "description": "地点名称"}
    #     }
    # },
    "location_search": {
        "description": "获取给定景点附近的相关关键字地点",
        "parameters": {
            "query": {"type": "string", "description": "检索关键字（如美食、银行...）"},
            "location": {"type": "string", "description": "地点经纬度(经度纬度之间用,分隔)"}
        }
    },
    "get_location_detail":{
        "description": "获取指定地点的详细信息",
        "parameters": {
            "uid": {"type": "string", "description": "利用location_search工具获得"}
        }
    },
    "get_location_distance":{
        "description":"获得给定地址名称列表之间的两两一组之间的步行距离。",
        "parameters":{
            "locations": {"type": "string", "description": "格式为 ['地点名称', '地点名称', '地点名称', ...]"}
        }
    }
}

# 工具调用模式的正则表达式 - 支持多个工具调用
TOOL_CALL_PATTERN = r'<tool_call>\s*({.*?})\s*</tool_call>'

def parse_tool_calls(text: str):
    """解析文本中的所有工具调用"""
    matches = re.findall(TOOL_CALL_PATTERN, text, re.DOTALL)
    tool_calls = []
    
    for match in matches:
        try:
            tool_call = json.loads(match)
            tool_calls.append(tool_call)
        except json.JSONDecodeError:
            continue
    
    return tool_calls

def execute_tool_calls(tool_calls: list) -> List[Dict]:
    """执行多个工具调用并返回结果列表"""
    results = []
    
    for tool_call in tool_calls:
        tool_name = tool_call.get("name")
        parameters = tool_call.get("parameters", {})
        print(tool_name, parameters)
        
        if not hasattr(TravelTools, tool_name):
            results.append({
                "name": tool_name,
                "status": "error",
                "result": f"未知工具 '{tool_name}'"
            })
            continue
        
        try:
            tool_func = getattr(TravelTools, tool_name)
            result = tool_func(**parameters)
            results.append({
                "name": tool_name,
                "status": "success",
                "result": result
            })
        except Exception as e:
            results.append({
                "name": tool_name,
                "status": "error",
                "result": f"执行错误: {str(e)}"
            })
    
    return results

def generate_with_tools(user_query: str, max_iterations: int = 100):
    """生成响应，支持多个工具调用"""
    messages = [{"role": "user", "content": user_query}]
    iterations = 0
    all_tool_results = []
    # 构建提示词，包含工具描述
    system_prompt = f"""你是一个旅行助手，你首先需要根据用户的需求，生成一些热门旅游景点，然后，选择调用的工具。
你可以使用以下工具：{TOOL_DESCRIPTIONS};"""+"""
请根据用户需求，选择合适的工具并严格按照以下格式调用：
<tool_call>
{"name": "工具名称", "parameters": {"参数名": "参数值"}}
</tool_call>
为获取旅游攻略，你需要获取天气，根据用户的需求筛选出热门景点，获取热门景点附近的酒店，获取景点两两之间的距离，检索景点附近的美食等。
"""
    while iterations < max_iterations:
        iterations += 1
        
        # 构造消息
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]
        
        # 应用聊天模板
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            do_sample=False,
        )
        
        # 准备模型输入
        model_inputs = tokenizer([text], return_tensors="ms", padding=True, truncation=True).to(torch_device)
        
        # 生成响应
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512,
        )
        
        # 处理生成的ID
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        # 解码响应
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(response)
        
        # 检查是否有工具调用
        tool_calls = parse_tool_calls(response)
        if not tool_calls:
            # 没有工具调用，返回最终响应
            return response, all_tool_results
        
        # 执行所有工具调用
        tool_results = execute_tool_calls(tool_calls)
        all_tool_results.extend(tool_results)
        
        # 将助手响应和工具结果添加到消息历史中
        messages.append({
            "role": "assistant", 
            "content": response
        })
        # if iterations == 0:
        #     system_prompt += "已经调用的工具即对应结果："
        for result in tool_results:
            result_str = f"{result['name']}: {result['status']} - {result['result']}"
            print(result_str)
            # system_prompt += result_str
            messages.append({
                "role": "tool", 
                "content": result_str
            })
    
    # 如果达到最大迭代次数，生成最终响应
    final_response = "我已经收集了以下信息：\n"
    for result in all_tool_results:
        final_response += f"- {result['name']}: {result['result']}\n"
    
    return final_response, all_tool_results

def get_tool_ans(city):
    # 获取景点列表
    result = {}
    attractions = json.loads(TravelTools.get_attractions(city))
    result['attractions'] = attractions
    poi_names = [poi['name'] for poi in attractions['pois']]
    poi_rating = [poi['biz_ext']['rating'] for poi in attractions['pois']]
    poi_spend = ['0.00' if isinstance(poi['biz_ext']['cost'], list) else poi['biz_ext']['cost'] for poi in attractions['pois']]
    poi_photo = []
    for poi in attractions['pois']:
        now = []
        for v in poi['photos']:
            now.append(v['url'])
        poi_photo.append(now)
    # 获取天气情况
    weather = TravelTools.get_all_weather(city)
    result['weather'] = weather
    # 获取景点周围的酒店、美食
    hotels = []
    foods = []
    for poi_name in poi_names[:2]:
        res = TravelTools.get_hotels(poi_name)
        res2 = TravelTools.get_foods(poi_name)
        hotels.append(f"{poi_name}附近酒店：{res}")
        foods.append(f"{poi_name}附近美食：{res2}")
        time.sleep(1)
    result['hotels'] = hotels
    result['foods'] = foods
    # 获取距离
    distances = TravelTools.get_location_distance(f'{poi_names}')
    # print(distances)
    simplifier = TourismDataSimplifier()
    result = simplifier.process_complete_data(result)
    return result, poi_names, distances, poi_rating, poi_spend, poi_photo

def get_small_transport(paths):
    # 计算两个地点之间 步行、骑行、驾车的时间花费
    lpoi = paths[0]
    ans = []
    for poi in paths[1:]:
        now = TravelTools.get_tow_poi_transport(lpoi, poi)
        ans.append(now)
        lpoi = poi
    return ans
        


# 注册MCP工具
@mcp.tool()
def get_attractions(city: str) -> str:
    return TravelTools.get_attractions(city)

@mcp.tool()
def get_hotels(district: str, checkin: str = None, checkout: str = None) -> str:
    return TravelTools.get_hotels(district, checkin, checkout)

@mcp.tool()
def get_red_blog(district: str) -> str:
    return TravelTools.get_red_blog(district)

@mcp.tool()
def get_route(origin: str, destination: str, tp: str = "driving") -> str:
    return TravelTools.get_route(origin, destination, tp)

@mcp.tool()
def get_all_weather(district: str) -> str:
    return TravelTools.get_all_weather(district)

@mcp.tool()
def get_location_code(locationName: str) -> str:
    return TravelTools.get_location_code(locationName)

@mcp.tool()
def location_search(query: str, location: str) -> str:
    return TravelTools.location_search(query, location)

@mcp.tool()
def get_location_detail(uid: str) -> str:
    return TravelTools.get_location_detail(uid)