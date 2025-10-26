from datetime import datetime
import json
import traceback
from flask import Blueprint, Response, stream_with_context
from flask import request
from flask import jsonify, make_response
import numpy as np
from . import model, tokenizer, torch_device, model_ppo, env
from app.utils.McpTool import generate_with_tools, get_tool_ans, get_small_transport
from app.utils.sample_tool import generate_unique_str
from app.ppo.get_ans import get_path
bp = Blueprint('main', __name__)

conversation_store: dict[str, list[dict]] = {}

@bp.route('/')
def home():
    return 'Hello, World!'

@bp.route('/api/getRespone', methods=['GET', 'POST'])
def getRespone():
    try:
        # 从请求中获取参数
        data = request.get_json()
        prompt = data.get('prompt', '')
        city = data.get('city', '')
        date = data.get('date', '')
        peolpe = data.get('people', '1人')
        tag = data.get('tag', '无')
        activityIntensity = data.get('activityIntensity', '2')
        activityIntensity = int(activityIntensity)+1
        money = data.get('money', 'comfort')
        food = data.get('food', '无')
        if activityIntensity == '1':
            activityIntensity = '每天两个景点'
        elif activityIntensity == '2':
            activityIntensity = '每天3个景点'
        elif activityIntensity == '3':
            activityIntensity = '每天四个景点'
        system_message = '你是一个有丰富经验的智能旅游规划助手，你需要根据用户的要求和各种偏好，使用提供的工具，生成详细的旅游规划推荐、天气出行建议（穿衣、雨雪天气优先选择室内）、美食建议、住宿建议。'
        user_message = f"帮我生成{date} {city} {peolpe}人 旅游攻略。个人兴趣：{tag}；期望活动强度：{activityIntensity}；期望金额预算：{money}；饮食偏好：{food}。必须满足的顾客的要求：{prompt}"
        print(user_message)
        # 随机生成对话id
        session_id = generate_unique_str(32)
        def generate(system_message):
            # 1. 调用MCP工具生成初步方案
            yield f"data: {json.dumps({'step': 'start', 'message': '开始生成旅游规划', 'process': 0}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'step': 'tool_processing', 'message': '正在调用MCP工具获取相关信息...', 'process': 20}, ensure_ascii=False)}\n\n"

            result, poi_names, distances, poi_rating, poi_spend, poi_photo = get_tool_ans(city)

            if not user_message:
                return jsonify({'error': 'No user message provided'}), 400
            if not city:
                return jsonify({'error': 'No city provided'}), 400
            # 2. 使用强化学习，根据景点间路程、花费、景点名称对方案进行调整
            yield f"data: {json.dumps({'step': 'tool_processing', 'message': '强化学习动态优化路线...', 'process': 40}, ensure_ascii=False)}\n\n"
            start_str, end_str = date.split(" 至 ")
            start_date = datetime.strptime(start_str, "%Y-%m-%d").date()
            end_date = datetime.strptime(end_str, "%Y-%m-%d").date()
            days = (end_date - start_date).days
            ppo_data = {
                'attractions': poi_names[:2],
                'distance_matrix': np.array(distances),
                'attraction_scores': np.array(poi_rating),
                'attraction_costs': np.array(poi_spend),
                'user_budget_type': 'economy',
                'min_attractions': 2
            }
            print(ppo_data)
            path = get_path(env, model_ppo, ppo_data)

            # 3. 返回调整后的旅游方案
            yield f"data: {json.dumps({'step': 'generate', 'message': '正在生成旅行方案...', 'process': 60}, ensure_ascii=False)}\n\n"
            # 规范数据返回格式
            system_message += '''你需要尽可能遵循强化学习给出的游玩顺序：'''+f"""{path["path"]},"""+'''按照用户给出的强度'''+f'{activityIntensity}'+'''（活动强度为几就是一天游玩几个景点）给出markdown结构化数据(包含名称、时间、门票价格等信息){
            "date": "2025-01-01",
            "time_begin": "13:00:00",
            "time_end": "16:00:00",
            "name": "北京海洋馆",
            "spend_money": "175",
            "category": "游玩",
            "detail": "一些详细信息和建议等"
            }, {
                "date": "2025-01-02",
                "time_begin": "09:00:00",
                "time_end": "12:00:00",
                "name": "故宫博物院",
                "spend_money": "100",
                "category": "游玩",
                "detail": "建议提前购票"
            }...; 
        '''

            # 构造消息
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
                {"role": "tool", "content": f'{result}'}
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
                max_new_tokens=1024,
            )
            
            # 处理生成的ID
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            # 解码响应
            responseS = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            print(responseS)


            # 4. 从response中提取json数据，格式化并添加图片、距离等信息
            imgs = []
            yield f"data: {json.dumps({'step': 'get_images', 'message': '正在获取景点图片...', 'process': 80}, ensure_ascii=False)}\n\n"
            # 筛选：poi_names中存在于responseS的元素（避免无效元素）
            valid_pois = [poi for poi in poi_names if poi in responseS]
            # 2. 排序：按valid_pois中元素在responseS的【首次出现顺序】排序
            # 原理：用str.find(poi)获取POI在responseS中的首次索引，按索引升序排列
            sorted_pois = sorted(valid_pois, key=lambda x: responseS.find(x))
            poi_to_photo = dict(zip(poi_names, poi_photo))
            # 按排序后的POI收集图片
            imgs = [poi_to_photo[poi] for poi in sorted_pois]

            # 5. 小交通路线规划
            yield f"data: {json.dumps({'step': 'get_transport', 'message': '正在获取景点间交通方式...', 'process': 90}, ensure_ascii=False)}\n\n"
            transport = get_small_transport(sorted_pois[:5])


            more_data = "# 更多信息\n\n"  # 一级标题规范，移除多余空行
            for i in range(min(len(sorted_pois), 5)):
                # 二级标题：景点名称，确保标题与内容间距规范
                now = f"## {poi_names[i]}\n\n"
                
                # 三级标题：景点图片，下方空行后引入图片
                now += "### 景点图片\n\n"
                # 图片引用：每张图片单独一行，确保 Markdown 图片语法正确
                for img in imgs[i]:
                    now += f"![{poi_names[i]}的景点照片]({img})\n\n"  # 图片描述补充景点名，增强可读性
                
                # 仅当不是最后一个景点时，添加交通方式推荐（当前循环仅1次，逻辑保留）
                if i != min(len(sorted_pois), 5) - 1:
                    now += f"### 推荐交通方式：{poi_names[i]} → {poi_names[i+1]}\n\n"
                    now += "> 推荐逻辑：综合耗时（权重70%）+ 花费（权重30%），选择总成本最低的出行方式\n\n"
                    
                    # 提取当前景点间的交通数据
                    now_tran = transport[i]
                    
                    # -------------------------- 1. 步行交通 --------------------------
                    walk_spend = float('inf') 
                    walk = now_tran[0] # 初始设为无穷大，便于后续比较
                    now += "#### 1. 步行\n\n"
                    if (
                        'result' in walk and
                        isinstance(walk['result'], dict) and
                        'routes' in walk['result'] and
                        isinstance(walk['result']['routes'], list) and
                        len(walk['result']['routes']) > 0 and
                        'distance' in walk['result']['routes'][0] and
                        'duration' in walk['result']['routes'][0]  # 补充duration字段校验，避免报错
                    ):
                        walk_dist = walk['result']['routes'][0]['distance']
                        walk_time = walk['result']['routes'][0]['duration'] // 60  # 转换为分钟
                        now += "- 距离：{} 米\n".format(walk_dist)
                        now += "- 耗时：{} 分钟\n".format(walk_time)
                        now += "- 花费：0 元\n\n"
                        # 计算步行综合成本（耗时权重70% + 花费权重30%）
                        walk_spend = (walk_time * 0.7) + (0 * 0.3)
                    else:
                        now += "- 暂无法获取步行路线数据\n\n"
                    
                    # -------------------------- 2. 驾车交通 --------------------------
                    drive_spend = float('inf')
                    drive = now_tran[1]
                    now += "#### 2. 驾车/打车\n\n"
                    if (
                        'result' in drive and
                        isinstance(drive['result'], dict) and
                        'routes' in drive['result'] and
                        isinstance(drive['result']['routes'], list) and
                        len(drive['result']['routes']) > 0 and
                        'distance' in drive['result']['routes'][0]
                    ):
                        drive_dist = drive['result']['routes'][0]['distance'] or "无"  # 处理空值
                        # 优化耗时计算（原逻辑可能有误，此处保留原公式，补充空值处理）
                        drive_time = (drive_dist / 60000 * 60) if drive_dist != "无" else "无"
                        drive_toll = drive['result']['routes'][0].get('toll', "无")  # 用get避免键不存在报错
                        now += "- 距离：{} 米\n".format(drive_dist)
                        now += "- 耗时：{} 分钟\n".format(round(drive_time, 1) if drive_time != "无" else drive_time)
                        now += "- 高速费：{} 元\n".format(drive_toll)
                        # 计算驾车综合成本（仅当数据有效时计算）
                        if drive_dist != "无" and drive_toll != "无":
                            drive_spend = (drive_time * 0.7) + (float(drive_toll) * 0.3)
                    else:
                        now += "- 暂无法获取驾车路线数据\n\n"
                    
                    # -------------------------- 3. 骑行交通 --------------------------
                    ride_spend = float('inf')
                    ride = now_tran[2]
                    now += "#### 3. 骑行\n\n"
                    if (
                        'result' in ride and
                        isinstance(ride['result'], dict) and
                        'routes' in ride['result'] and
                        isinstance(ride['result']['routes'], list) and
                        len(ride['result']['routes']) > 0 and
                        'distance' in ride['result']['routes'][0] and
                        'duration' in ride['result']['routes'][0]
                    ):
                        ride_dist = ride['result']['routes'][0]['distance']
                        ride_time = ride['result']['routes'][0]['duration'] // 60
                        now += "- 距离：{} 米\n".format(ride_dist)
                        now += "- 耗时：{} 分钟\n".format(ride_time)
                        now += "- 花费：1.5 元\n\n"
                        # 计算骑行综合成本
                        ride_spend = (ride_time * 0.7) + (1.5 * 0.3)
                    else:
                        now += "- 暂无法获取骑行路线数据\n\n"
                    
                    # -------------------------- 4. 公共交通（公交/地铁） --------------------------
                    transit_spend = float('inf')
                    transit = now_tran[3]
                    now += "#### 4. 公交/地铁\n\n"
                    if (
                        'result' in transit and
                        isinstance(transit['result'], dict) and
                        'routes' in transit['result'] and
                        isinstance(transit['result']['routes'], list) and
                        len(transit['result']['routes']) > 0 and
                        'distance' in transit['result']['routes'][0] and
                        'price' in transit['result']['routes'][0]  # 补充price字段校验
                    ):
                        transit_dist = transit['result']['routes'][0]['distance']
                        # 优化耗时计算（原逻辑可能有误，此处保留原公式）
                        transit_time = transit_dist // 60
                        transit_price = transit['result']['routes'][0]['price']
                        now += "- 距离：{} 米\n".format(transit_dist)
                        now += "- 耗时：{} 分钟\n".format(transit_time)
                        now += "- 花费：{} 元\n\n".format(transit_price)
                        # 计算公共交通综合成本
                        transit_spend = (transit_time * 0.7) + (float(transit_price) * 0.3)
                    else:
                        now += "- 暂无法获取公共交通路线数据\n\n"
                    
                    # -------------------------- 交通方式推荐 --------------------------
                    now += "### 最终推荐\n\n"
                    # 过滤无效成本（排除无穷大的情况，即路线数据缺失的方式）
                    valid_spends = {
                        "步行": walk_spend,
                        "骑行": ride_spend,
                        "驾车/打车": drive_spend,
                        "公共交通": transit_spend
                    }
                    valid_spends = {k: v for k, v in valid_spends.items() if v != float('inf')}
                    
                    if not valid_spends:
                        now += "⚠️  所有交通方式数据缺失，无法推荐\n\n"
                    else:
                        # 原逻辑：若驾车成本≥50优先推荐，否则选成本最低的方式
                        if drive_spend != float('inf') and drive_spend >= 50:
                            now += "**推荐选择：驾车/打车**\n\n"
                            now += "> 推荐理由：驾车综合成本符合优先推荐阈值（≥50）\n\n"
                        else:
                            min_spend_type = min(valid_spends, key=valid_spends.get)
                            now += "**推荐选择：{}**\n\n".format(min_spend_type)
                            now += "> 推荐理由：综合耗时（70%）+ 花费（30%）计算，该方式总成本最低（成本值：{:.2f}）\n\n".format(valid_spends[min_spend_type])
                
                # 将当前景点的完整信息追加到总数据中
                more_data += now



            # 记录聊天历史记录（因模型限制，仅保留最近3轮， 可自行调整）
            if conversation_store.get(session_id) is None:
                conversation_store[session_id] = []
            conversation_store[session_id].append({
                "user_message": user_message,
                "tool_message": result,
                "response": responseS,
            })
            global response_data
            response_data = {
                # 'system_message': system_message,
                # 'user_message': user_message,
                "tool_message": result,
                'message': responseS,
                "poi_name": sorted_pois,
                'imgs': imgs,
                "session_id": session_id,
                "transport": transport,
                "more_message": more_data,
            }
            yield f"data: {json.dumps(response_data, ensure_ascii=False)}\n\n"
            # yield f"data: {json.dumps({'step': 'end', 'message': '生成完成', 'process': 100}, ensure_ascii=False)}\n\n"
        
        # 返回结果
        # json_str = json.dumps(response_data, ensure_ascii=False)
        # response = make_response(json_str)
        # response.headers['Content-Type'] = 'application/json; charset=utf-8'
        # return response
        return Response(
            stream_with_context(generate(system_message)),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type',
            }
        )
    
    except Exception as e:
        print(str(e))
        print(traceback.format_exc())
        
        def generate_error(e):
            yield f"data: {json.dumps({'step': 'error', 'error': str(e)}, ensure_ascii=False)}\n\n"
        
        return Response(
            stream_with_context(generate_error(e)),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type',
            }
        )
    
@bp.route('/api/moreChat', methods=['GET', 'POST'])
def moreChat():
    try:
        # 从请求中获取参数
        data = request.get_json()
        user_message = data.get("user_message", '')
        session_id = data.get("session_id", '')
        activityIntensity = data.get('activityIntensity', '中等')
        activityIntensity = int(activityIntensity)+1
        system_message = '你是一个有丰富经验的智能旅游规划助手，你需要根据用户的要求和各种偏好，结合历史记录中给出的工具生成的景点详细信息按照用户的要求对旅游攻略进行调整。'

        # 规范数据返回格式
        system_message += f'''你需要尽可能遵循用户的偏好要求，结合给出的强度{activityIntensity}（活动强度为几就是一天游玩几个景点）给出json结构化数据(包含名称、时间、门票价格等信息)，格式：{
        "date": "2025-01-01",
        "time_begin": "13:00:00",
        "time_end": "16:00:00",
        "name": "北京海洋馆",
        "spend_money": "175",
        "category": "游玩",
        "detail": "一些详细信息和建议等"
        }, {
            "date": "2025-01-02",
            "time_begin": "09:00:00",
            "time_end": "12:00:00",
            "name": "故宫博物院",
            "spend_money": "100",
            "category": "游玩",
            "detail": "建议提前购票"
        }...; 
    '''

        # 构造消息， 结合历史消息
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
            {"history": "history", "content": f'{conversation_store[session_id]}'}
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
            max_new_tokens=1024,
        )
        
        # 处理生成的ID
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        # 解码响应
        responseS = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # 记录聊天历史记录（因模型限制，仅保留最近3轮， 可自行调整）
        if conversation_store.get(session_id) is None:
            conversation_store[session_id] = []
        conversation_store[session_id].append({
            "user_message": user_message,
            "response": responseS,
        })
        if len(conversation_store[session_id]) > 4:
            conversation_store[session_id] = conversation_store[:1].extend(conversation_store[2:])
        print(conversation_store[session_id])
        
        # 返回结果
        response_data = {
            # 'system_message': system_message,
            # 'user_message': user_message,
            # "tool_message": result,
            'response': responseS,
            "session_id": session_id
        }

        json_str = json.dumps(response_data, ensure_ascii=False)
        response = make_response(json_str)
        response.headers['Content-Type'] = 'application/json; charset=utf-8'
        return response
    
    except Exception as e:
        print(str(e))
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500