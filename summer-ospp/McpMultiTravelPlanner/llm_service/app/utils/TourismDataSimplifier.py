import json
import time
from typing import List, Dict, Tuple

class TourismDataSimplifier:
    """旅游数据精简器"""
    
    def __init__(self):
        self.simplified_data = {}
    
    def simplify_attractions(self, attractions_data: Dict) -> List[Dict]:
        """精简景点信息"""
        simplified = []
        pois = attractions_data.get('pois', [])
        for attraction in pois[:8]:  # 限制数量
            simplified_attraction = {
                '名称': attraction.get('name', ''),
                '地址': attraction.get('address', ''),
                '等级': attraction.get('biz_ext', {}).get('level', '未知'),
                '评分': attraction.get('biz_ext', {}).get('rating', '未知'),
                '开放时间': self._simplify_time(attraction.get('biz_ext', {}).get('open_time', '')),
                '电话': attraction.get('tel', '')[:15] if attraction.get('tel') else '',
                '类型': attraction.get('type', '未知'),

            }
            # 移除空值
            simplified_attraction = {k: v for k, v in simplified_attraction.items() if v}
            simplified.append(simplified_attraction)
        return simplified
    
    def simplify_weather(self, weather_data: str) -> Dict:
        """精简天气信息"""
        try:
            # 处理unicode转义并解析JSON
            weather_str = weather_data.encode('utf-8').decode('unicode_escape')
            weather_dict = json.loads(weather_str)
            weather_data = weather_dict.get('result', {})
        except:
            return {}
        
        # 当前天气
        current = weather_data.get('now', {})
        simplified_weather = {
            '当前天气': current.get('text', ''),
            '温度': f"{current.get('temp', '')}°C",
            '体感温度': f"{current.get('feels_like', '')}°C",
            '湿度': f"{current.get('rh', '')}%",
            '风速': f"{current.get('wind_class', '')} {current.get('wind_dir', '')}"
        }
        
        # 未来3天预报
        forecasts = weather_data.get('forecasts', [])[:3]
        forecast_list = []
        for forecast in forecasts:
            forecast_list.append({
                '日期': forecast.get('date', ''),
                '白天': forecast.get('text_day', ''),
                '夜间': forecast.get('text_night', ''),
                '温度': f"{forecast.get('low', '')}~{forecast.get('high', '')}°C"
            })
        
        if forecast_list:
            simplified_weather['未来预报'] = forecast_list
        
        return simplified_weather
    
    def simplify_hotels(self, hotels_data: List) -> List[Dict]:
        """精简酒店信息"""
        simplified_hotels = []
        for hotel_group in hotels_data:
            position = hotel_group.find("：")
            hotel_group = hotel_group[position+1:]
            if isinstance(hotel_group, str):
                try:
                    # 尝试解析字符串格式的酒店数据
                    hotel_group = json.loads(hotel_group.replace("'", '"'))
                except:
                    continue
            
            if isinstance(hotel_group, list):
                for hotel in hotel_group:  # 每个区域取前3个
                    if isinstance(hotel, dict) and hotel.get('status') == 0 and 'result' in hotel:
                        result = hotel['result']
                        simplified_hotel = {
                            '名称': result.get('name', ''),
                            '地址': result.get('address', ''),
                            '等级': result.get('detail_info', {}).get('level', ''),
                            '评分': result.get('detail_info', {}).get('overall_rating', ''),
                            '电话': result.get('telephone', '')[:15] if result.get('telephone') else ''
                        }
                        simplified_hotel = {k: v for k, v in simplified_hotel.items() if v}
                        if simplified_hotel:
                            simplified_hotels.append(simplified_hotel)
        
        return simplified_hotels[:10]  # 限制总数
    
    def simplify_food(self, food_data: List) -> List[Dict]:
        """精简美食信息"""
        simplified_food = []
        
        for food_group in food_data:
            position = food_group.find("：")
            food_group = food_group[position+1:]
            if isinstance(food_group, str):
                try:
                    # 处理美食数据
                    if "天配额超限" in food_group:
                        continue  # 跳过配额限制的数据
                    # 尝试解析JSON
                    food_data_parsed = json.loads(food_group.replace("'", '"'))
                except:
                    continue
            else:
                food_data_parsed = food_group
            
            if isinstance(food_data_parsed, dict) and 'results' in food_data_parsed:
                for restaurant in food_data_parsed['results']:
                    simplified_restaurant = {
                        '名称': restaurant.get('name', ''),
                        '地址': restaurant.get('address', ''),
                        '电话': restaurant.get('telephone', '')[:15] if restaurant.get('telephone') else ''
                    }
                    simplified_restaurant = {k: v for k, v in simplified_restaurant.items() if v}
                    if simplified_restaurant:
                        simplified_food.append(simplified_restaurant)
        
        return simplified_food[:8]  # 限制总数
    
    def _simplify_time(self, time_str: str) -> str:
        """简化时间描述"""
        if not time_str:
            return ""
        # 移除冗余描述，保留核心时间
        import re
        time_str = re.sub(r'最晚进入\d{1,2}:\d{2}', '', time_str)
        time_str = re.sub(r'停止售票', '', time_str)
        return time_str.strip('; ')
    
    def process_complete_data(self, raw_data: Dict) -> Dict:
        """处理完整数据"""
        simplified = {}
        
        # 景点信息
        if 'attractions' in raw_data:
            simplified['景点'] = self.simplify_attractions(raw_data['attractions'])
        
        # 天气信息
        if 'weather' in raw_data:
            simplified['天气'] = self.simplify_weather(raw_data['weather'])
        
        # 酒店信息
        if 'hotels' in raw_data:
            simplified['酒店'] = self.simplify_hotels(raw_data['hotels'])
        
        # 美食信息
        if 'foods' in raw_data:
            simplified['美食'] = self.simplify_food(raw_data['foods'])
        
        return simplified