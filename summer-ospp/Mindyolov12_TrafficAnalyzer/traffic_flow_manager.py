import cv2
import numpy as np
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
from collections import deque
import time
import math

@dataclass
class TrafficFlowStats:
    """交通流量分析的统计数据"""
    lane_id: int
    entry_count: int = 0
    exit_count: int = 0
    current_flow: int = 0
    avg_transit_time: float = 0.0
    transit_times: deque = None
    
    def __post_init__(self):
        if self.transit_times is None:
            self.transit_times = deque(maxlen=50)

class TrafficFlowManager:
    def __init__(self, video_width: int, video_height: int):
        """初始化交通流量管理器"""
        self.video_width = video_width
        self.video_height = video_height
        self.lane_stats: Dict[int, TrafficFlowStats] = {}
        
        # 车辆追踪
        self.vehicles_in_scene: Set[int] = set()  # 当前场景中的车辆ID
        self.vehicle_entry_times: Dict[int, Dict] = {}  # 每辆车的进入时间和地点
        self.vehicle_exit_times: Dict[int, Dict] = {}  # 退出时间和地点 
        
        # 进出区域
        self.entry_zones: Dict[int, np.ndarray] = {}
        self.exit_zones: Dict[int, np.ndarray] = {}
        
        # 交通流量阈值
        self.traffic_flow_levels = {
            "Low": (0, 5),      # 每分钟0-5辆车
            "Moderate": (5, 15), # 每分钟5至15辆车
            "High": (15, float('inf'))  # 每分钟超过15辆车
        }
        
    def initialize_zones(self, lanes: Dict[int, Dict]):
        """根据车道配置初始化出入口区域"""
        for lane_id, lane_info in lanes.items():
            polygon = lane_info['polygon']
            
            # 在车道顶部创建入口区域
            entry_points = np.array([
                polygon[0],  # 左上角
                polygon[1],  # 右上角
                [polygon[1][0], polygon[1][1] + 100],  # 右下角
                [polygon[0][0], polygon[0][1] + 100]   # 左下角
            ], dtype=np.int32)
            
            # 在泳道底部创建退出区域
            exit_points = np.array([
                [polygon[3][0], polygon[3][1] - 100],  # 左上角
                [polygon[2][0], polygon[2][1] - 100],  # 右上角
                polygon[2],  # 右下角
                polygon[3]   # 左下角
            ], dtype=np.int32)
            
            self.entry_zones[lane_id] = entry_points
            self.exit_zones[lane_id] = exit_points
            
            # 初始化每条车道的统计数据
            self.lane_stats[lane_id] = TrafficFlowStats(lane_id=lane_id)
    
    def check_vehicle_zone(self, vehicle_position: Tuple[int, int], lane_id: int) -> Tuple[bool, bool]:
        """检查车辆是否在进入或离开区域"""
        in_entry = False
        in_exit = False
        
        if lane_id in self.entry_zones:
            in_entry = cv2.pointPolygonTest(self.entry_zones[lane_id], vehicle_position, False) >= 0
            
        if lane_id in self.exit_zones:
            in_exit = cv2.pointPolygonTest(self.exit_zones[lane_id], vehicle_position, False) >= 0
            
        return in_entry, in_exit
    
    def process_vehicle(self, track_id: int, position: Tuple[int, int], lane_id: int, current_time: float):
        """处理车辆以进行交通流量分析"""
        if lane_id is None:
            return
        
        # 获取车辆位置
        in_entry, in_exit = self.check_vehicle_zone(position, lane_id)
        
        # 车辆进入车道
        if in_entry and track_id not in self.vehicles_in_scene:
            self.vehicles_in_scene.add(track_id)
            self.vehicle_entry_times[track_id] = {
                'time': current_time,
                'lane_id': lane_id,
                'position': position
            }
            self.lane_stats[lane_id].entry_count += 1
            self.lane_stats[lane_id].current_flow += 1
        
        # 车辆偏离车道
        elif in_exit and track_id in self.vehicles_in_scene:
            if track_id in self.vehicle_entry_times:
                entry_data = self.vehicle_entry_times[track_id]
                entry_lane = entry_data['lane_id']
                
                # 仅当从同一车道驶出时才计数
                if entry_lane == lane_id:
                    transit_time = current_time - entry_data['time']
                    
                    # 更新车道统计数据
                    self.lane_stats[lane_id].exit_count += 1
                    self.lane_stats[lane_id].current_flow -= 1
                    self.lane_stats[lane_id].transit_times.append(transit_time)
                    
                    # 更新平均运输时间
                    times = self.lane_stats[lane_id].transit_times
                    self.lane_stats[lane_id].avg_transit_time = sum(times) / len(times) if times else 0
                    
                # 存储退出数据
                self.vehicle_exit_times[track_id] = {
                    'time': current_time,
                    'lane_id': lane_id,
                    'position': position,
                    'transit_time': current_time - entry_data['time']
                }
                
            # 从跟踪中移除
            self.vehicles_in_scene.discard(track_id)
    
    def calculate_speed_from_transit(self, transit_time: float, lane_length_meters: float = 100.0) -> float:
        """Calculate speed from transit time and lane length"""
        if transit_time <= 0:
            return 0.0
        
        # 速度 = 距离 / 时间，转换为公里/小时
        return (lane_length_meters / transit_time) * 3.6
    
    def estimate_realistic_speed(self, lane_id: int, position_y: int) -> float:
        """Generate realistic speed based on lane, position and traffic conditions"""
        # 不同车道类型的基础速度范围
        # 将车道调整为7个
        base_speeds = {
            0: (5, 40),   # 应急车道：减速至中低速
            1: (30, 60),  # 慢车道
            2: (40, 80),  # 中间车道
            3: (60, 100),  # 快车道
            4: (60, 100),  # 快车道
            5: (40, 80),  # 中间车道
            # 6: (30, 60), # 慢车道
            6: (5, 40),   # 应急车道：减速至中低速
        }
        
        # 默认速度范围
        min_speed, max_speed = base_speeds.get(lane_id, (30, 80))
        
        # 根据交通拥堵情况调整
        if lane_id in self.lane_stats:
            flow = self.lane_stats[lane_id].current_flow
            if flow > 15:  # 交通拥堵
                min_speed = max(min_speed * 0.4, 10)
                max_speed = min_speed + 20
            elif flow > 8:  # 中等交通流量
                min_speed = min_speed * 0.7
                max_speed = max_speed * 0.8
        
        # 根据位置微调速度（入口/出口处较慢）
        position_factor = abs((self.video_height/2 - position_y) / (self.video_height/2))
        position_adjustment = 1.0 - (0.2 * (1.0 - position_factor))
        
        # 添加一些小的随机性
        random_factor = np.random.uniform(0.85, 1.15)
        
        final_speed = min_speed + (max_speed - min_speed) * position_factor * position_adjustment * random_factor
        return round(min(max(final_speed, 5), 120), 1)  # 夹紧速度介于5至120公里/小时之间
    
    def get_flow_level(self, lane_id: int, time_window: float = 60.0) -> str:
        """Get traffic flow level for a lane"""
        if lane_id not in self.lane_stats:
            return "Unknown"
            
        # 计算流量（车辆数/分钟）
        count = self.lane_stats[lane_id].entry_count
        flow_rate = (count / time_window) * 60
        
        # 确定流量水平
        for level, (min_flow, max_flow) in self.traffic_flow_levels.items():
            if min_flow <= flow_rate < max_flow:
                return level
                
        return "Moderate"

    def draw_zones(self, frame: np.ndarray) -> np.ndarray:
        """在框架上绘制入口和出口参考线"""
        # 绘制入口参考线（顶部绿色水平线）
        y_entry = 100  # 入口线在画面上方
        cv2.line(frame, (0, y_entry), (self.video_width, y_entry), (0, 255, 0), 2)
        cv2.putText(frame, "Entry Line", (1000, y_entry - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 绘制出口参考线（底部红色水平线）
        y_exit = self.video_height - 100  # 出口线在画面下方
        cv2.line(frame, (0, y_exit), (self.video_width, y_exit), (0, 0, 255), 2)
        cv2.putText(frame, "Exit Line", (1000, y_exit + 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return frame
    
    def draw_flow_stats(self, frame: np.ndarray, start_y: int = 250) -> np.ndarray:
        """在帧上绘制交通流量统计"""
        y_offset = start_y
        
        cv2.putText(frame, "TRAFFIC FLOW STATS:", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 25
        
        for lane_id, stats in self.lane_stats.items():
            flow_text = f"Lane {lane_id}: IN:{stats.entry_count} OUT:{stats.exit_count} "
            flow_text += f"CURRENT:{stats.current_flow} "
            
            if stats.transit_times:
                flow_text += f"AVG TIME:{stats.avg_transit_time:.1f}s"
                
            flow_level = self.get_flow_level(lane_id)
            
            # 基于流量水平的颜色编码
            color = (0, 255, 0)  # 绿色表示低
            if flow_level == "High":
                color = (0, 0, 255)  # 红色代表高
            elif flow_level == "Moderate":
                color = (0, 165, 255)  # 橙色代表温和
                
            cv2.putText(frame, flow_text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            cv2.putText(frame, f"FLOW: {flow_level}", (400, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                       
            y_offset += 20
            
        return frame
    
