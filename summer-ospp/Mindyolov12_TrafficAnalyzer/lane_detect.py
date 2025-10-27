import json
import cv2
import numpy as np
import time
import os
import sys
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict, deque
from traffic_flow_manager import TrafficFlowManager
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import mindspore as ms
from mindspore import Tensor, nn
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
from mindyolo.models import create_model
from mindyolo.utils import logger
from mindyolo.utils.config import load_config, Config
from demo.predict import detect
from mindyolo.data import COCO80_TO_COCO91_CLASS
import random

def draw_result_on_frame(frame: np.ndarray, result_dict: Dict, data_names: List[str], is_coco_dataset: bool = True) -> np.ndarray:
    """
    在视频帧上绘制检测框与标签
    """
    im = frame.copy()
    category_id = result_dict.get("category_id", [])
    bbox = result_dict.get("bbox", [])
    score = result_dict.get("score", [])
    seg = result_dict.get("segmentation", None)
    mask = None if seg is None else np.zeros_like(im, dtype=np.float32)

    for i in range(len(bbox)):
        # draw box
        x_l, y_t, w, h = bbox[i][:]
        x_r, y_b = x_l + w, y_t + h
        x_l, y_t, x_r, y_b = int(x_l), int(y_t), int(x_r), int(y_b)
        _color = [random.randint(0, 255) for _ in range(3)]
        cv2.rectangle(im, (x_l, y_t), (x_r, y_b), tuple(_color), 2)
        if seg:
            _color_seg = np.array([random.randint(0, 255) for _ in range(3)], np.float32)
            mask += seg[i][:, :, None] * _color_seg[None, None, :]

        # draw label
        if is_coco_dataset:
            # 将 COCO91 的类别 id 映射回 data_names 的索引
            class_name_index = COCO80_TO_COCO91_CLASS.index(category_id[i]) if category_id[i] in COCO80_TO_COCO91_CLASS else 0
        else:
            class_name_index = category_id[i]
        class_name_index = min(max(class_name_index, 0), len(data_names) - 1)
        class_name = data_names[class_name_index]
        text = f"{class_name}: {score[i]:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(im, (x_l, y_t - text_h - baseline), (x_l + text_w, y_t), tuple(_color), -1)
        cv2.putText(im, text, (x_l, y_t - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    if seg:
        im = (0.7 * im + 0.3 * mask).astype(np.uint8)
    return im

# 记录车辆信息
@dataclass
class VehicleInfo:
    track_id: int
    vehicle_type: str
    speed: float
    lane_id: int
    entry_time: float
    last_position: Tuple[int, int]
    positions_history: deque
    estimated_speed: float = 0.0  # 估计的真实速度
    speed_timestamp: float = 0.0  # 上次速度更新时间
    # 可疑行为
    suspicious_behaviors: List[str] = None
    suspicious_score: float = 0.0

class SimpleBYTETracker:
    """ByteTrack追踪算法"""
    def __init__(self, max_time_lost=30):
        self.tracks = {}
        self.track_id_count = 0
        self.max_time_lost = max_time_lost
        
    def update(self, detections):
        """更新跟踪器"""
        if len(detections) == 0:
            # 增加所有现有跟踪的丢失时间
            for track in self.tracks.values():
                track['time_since_update'] += 1
            
            # 移除长时间丢失的跟踪
            tracks_to_remove = []
            for track_id, track in self.tracks.items():
                if track['time_since_update'] > self.max_time_lost:
                    tracks_to_remove.append(track_id)
            
            for track_id in tracks_to_remove:
                del self.tracks[track_id]
            
            return []
        
        tracked_detections = []
        
        # 基于距离的匹配
        for det in detections:
            bbox = det['bbox']
            center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
            
            best_match = None
            best_distance = float('inf')
            
            # 寻找最近的现有跟踪
            for track_id, track in self.tracks.items():
                if track['time_since_update'] > 0:  # 只考虑丢失的跟踪
                    distance = np.sqrt((center[0] - track['last_pos'][0])**2 + 
                                     (center[1] - track['last_pos'][1])**2)
                    if distance < best_distance and distance < 100:  # 距离阈值
                        best_distance = distance
                        best_match = track_id
            
            if best_match is not None:
                # 更新现有跟踪
                track_id = best_match
                self.tracks[track_id]['last_pos'] = center
                self.tracks[track_id]['time_since_update'] = 0
                self.tracks[track_id]['bbox'] = bbox
            else:
                # 创建新跟踪
                track_id = self.track_id_count
                self.track_id_count += 1
                self.tracks[track_id] = {
                    'last_pos': center,
                    'time_since_update': 0,
                    'bbox': bbox
                }
            
            tracked_det = det.copy()
            tracked_det['track_id'] = track_id
            tracked_detections.append(tracked_det)
        
        # 增加未匹配跟踪的丢失时间
        matched_track_ids = {det['track_id'] for det in tracked_detections}
        for track_id in self.tracks:
            if track_id not in matched_track_ids:
                self.tracks[track_id]['time_since_update'] += 1
        
        return tracked_detections

class LaneDetector:
    def __init__(self, config_path: str, video_width: int, video_height: int):
        """
        使用JSON文件中的多边形数据初始化车道检测器
        """
        self.video_width = video_width
        self.video_height = video_height
        self.lanes = {}
        self.emergency_lane_id = None
        self.load_polygon_data(config_path)
        
    def load_polygon_data(self, config_path: str):
        """从JSON文件加载并缩放多边形数据"""
        try:
            print(f"正在从JSON文件加载车道配置: {config_path}")
            
            if not os.path.exists(config_path):
                print(f"错误: 车道配置文件不存在: {config_path}")
                self._create_default_lanes()
                return
            
            with open(config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 从配置中获取原始视频尺寸
            original_width = data.get('video_width', 1920)
            original_height = data.get('video_height', 1080)
            
            print(f"配置文件中的视频尺寸: {original_width}x{original_height}")
            print(f"当前视频尺寸: {self.video_width}x{self.video_height}")
            
            # 计算缩放比例
            scale_x = self.video_width / original_width
            scale_y = self.video_height / original_height
            
            print(f"缩放比例: X={scale_x:.3f}, Y={scale_y:.3f}")
            
            # 加载并缩放车道多边形
            lanes_data = data.get('lanes', [])
            if not lanes_data:
                print("警告: JSON配置文件中没有找到车道数据")
                self._create_default_lanes()
                return
            
            for lane_data in lanes_data:
                lane_id = lane_data['id']
                points = np.array(lane_data['points'])
                
                # 缩放点以匹配当前视频分辨率
                scaled_points = points.copy().astype(float)
                # if scale_x != 1.0 or scale_y != 1.0:
                #     scaled_points[:, 0] *= scale_x
                #     scaled_points[:, 1] *= scale_y
                
                self.lanes[lane_id] = {
                    'polygon': scaled_points.astype(np.int32),
                    'is_emergency': lane_data.get('is_emergency', False),
                    'vehicle_count': 0,
                    'speeds': deque(maxlen=50),
                    'congestion_level': 'Normal'
                }
                
                if lane_data.get('is_emergency', False):
                    self.emergency_lane_id = lane_id
                    print(f"车道 {lane_id} 标记为应急车道")
            
            print(f"成功加载 {len(self.lanes)} 条车道配置")
                    
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            print("使用默认车道配置")
            self._create_default_lanes()
        except Exception as e:
            print(f"加载车道配置时发生错误: {e}")
            print("使用默认车道配置")
            self._create_default_lanes()
    
    def _create_default_lanes(self):
        """如果JSON文件不可用，则创建默认车道"""
        lane_width = self.video_width // 4
        
        for i in range(4):
            x_start = i * lane_width
            x_end = (i + 1) * lane_width
            
            polygon = np.array([
                [x_start, 0],
                [x_end, 0],
                [x_end, self.video_height],
                [x_start, self.video_height]
            ])
            
            self.lanes[i] = {
                'polygon': polygon,
                'is_emergency': i == 0,  # 第一条车道是应急车道
                'vehicle_count': 0,
                'speeds': deque(maxlen=50),
                'congestion_level': 'Normal'
            }
            
            if i == 0:
                self.emergency_lane_id = i
    
    def get_vehicle_lane(self, bbox: Tuple[int, int, int, int]) -> Optional[int]:
        """根据车辆的边界框确定其所在车道"""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        for lane_id, lane_info in self.lanes.items():
            if cv2.pointPolygonTest(lane_info['polygon'], (center_x, center_y), False) >= 0:
                return lane_id
        
        return None
    
    def update_lane_stats(self, lane_id: int, speed: float):
        """使用新车辆数据更新车道统计信息"""
        if lane_id in self.lanes:
            self.lanes[lane_id]['speeds'].append(speed)
            self._update_congestion_level(lane_id)
    
    def _update_congestion_level(self, lane_id: int):
        """根据车辆数量和平均速度更新拥堵等级"""
        lane = self.lanes[lane_id]
        vehicle_count = lane['vehicle_count']
        avg_speed = np.mean(lane['speeds']) if lane['speeds'] else 0
        
        if vehicle_count > 10 and avg_speed < 30:
            lane['congestion_level'] = 'Heavy'
        elif vehicle_count > 5 and avg_speed < 50:
            lane['congestion_level'] = 'Moderate'
        else:
            lane['congestion_level'] = 'Normal'
    
    def get_average_speed(self, lane_id: int) -> float:
        """获取特定车道的平均速度"""
        if lane_id in self.lanes and self.lanes[lane_id]['speeds']:
            return np.mean(self.lanes[lane_id]['speeds'])
        return 0.0

class LaneVehicleProcessor:
    def __init__(self, config_path: str, weight_path: str, lane_config_path: str):
        """初始化MindSpore YOLO模型"""
        self.config_path = config_path
        self.weight_path = weight_path
        self.lane_config_path = lane_config_path
        
        # 初始化MindSpore环境
        ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend")  # 可以改为CPU或Ascend
        
        # 加载模型配置
        self._load_model_config(self.config_path)
        
        # 根据数据集名称判断是否为 COCO（影响类别 id 映射与可视化）
        self.is_coco_dataset = True
        try:
            ds_name = getattr(self.model_config.data, 'dataset_name', '')
            if isinstance(ds_name, str) and ds_name:
                self.is_coco_dataset = ('coco' in ds_name.lower())
        except Exception:
            self.is_coco_dataset = True
        
        # 创建模型（若权重为空或不存在，则跳过加载）
        ckpt_path = self.weight_path if (self.weight_path and os.path.exists(self.weight_path)) else None
        self.network = create_model(
            model_name=self.model_config.network.model_name,
            model_cfg=self.model_config.network,
            num_classes=self.model_config.data.nc,
            sync_bn=False,
            checkpoint_path=ckpt_path,
        )
        self.network.set_train(False)
        
        # 车辆追踪器
        self.tracker = SimpleBYTETracker()
        self.lane_detector = None
        
        # 车辆追踪数据
        self.tracked_vehicles: Dict[int, VehicleInfo] = {}
        # 保存数据集的类别名，供过滤/显示使用
        self.data_names = list(self.model_config.data.names)
        self.vehicle_types = {
            # 使用 COCO91 索引：1=person, 2=bicycle, 3=car, 4=motorcycle, 6=bus, 8=truck
            3: 'car',
            4: 'motorcycle',
            6: 'bus',
            8: 'truck',
            2: 'bicycle',
            1: 'person'
        }
        
        # 应急车道违规行为
        self.emergency_violations = []

        # 添加流量管理器
        self.flow_manager = None
        
        # 统计
        self.start_time = time.time()
        self.frame_count = 0

        # 可疑车辆追踪
        self.suspicious_vehicles: Dict[int, VehicleInfo] = {}
        self.suspicious_threshold = 3.0  # 可疑行为评分阈值

        # 车辆类别统计
        self.vehicle_count_stats = {
            'car': 0,
            'truck': 0, 
            'bus': 0,
            'motorcycle': 0,
            'bicycle': 0,
            'person': 0,
            'unknown': 0
        }
        
        # 按车道分类的车辆统计
        self.lane_vehicle_stats = {}
        
    def _load_model_config(self, config_path: str):
        """加载模型配置"""
        cfg_dict, _, _ = load_config(config_path)
        self.model_config = Config(cfg_dict)
        
    def initialize_video_processing(self, video_path: str):
        """初始化视频处理与车道检测器"""
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        self.lane_detector = LaneDetector(self.lane_config_path, width, height)

        # 初始化交通流管理器
        self.flow_manager = TrafficFlowManager(width, height)
        self.flow_manager.initialize_zones(self.lane_detector.lanes)
        
        return width, height
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """使用MindSpore YOLO检测与跟踪处理单帧图像"""
        # 更新统计信息
        self.frame_count += 1
        current_time = time.time()
        
        # YOLO检测
        # print(f"[DEBUG] conf_free={getattr(self.model_config, 'conf_free', False)}, stride={self.model_config.network.stride}, nc={self.model_config.data.nc}, is_coco={self.is_coco_dataset}")
        result_dict = detect(
            network=self.network,
            img=frame,
            conf_thres=0.25,
            iou_thres=0.65,
            conf_free=getattr(self.model_config, 'conf_free', False),
            exec_nms=True,
            nms_time_limit=60.0,
            img_size=640,
            stride=max(max(self.model_config.network.stride), 32),
            num_class=self.model_config.data.nc,
            is_coco_dataset=self.is_coco_dataset
        )
        # print(f"[DEBUG] detections={len(result_dict.get('bbox', []))}")
        
        # 先把检测框直接画到帧上，便于立即可视化
        frame = draw_result_on_frame(frame, result_dict, self.model_config.data.names, is_coco_dataset=self.is_coco_dataset)
        
        # 筛选车辆检测结果
        vehicle_detections = self._filter_vehicle_detections(result_dict)
        
        # 车辆跟踪
        tracked_detections = self.tracker.update(vehicle_detections)
        
        # 处理追踪到的车辆
        self._process_tracked_vehicles(tracked_detections, current_time)
        
        # 绘制可视化图表（在已画有检测框的帧上继续叠加）
        annotated_frame = self._draw_annotations(frame, tracked_detections)
        
        return annotated_frame
    
    def _filter_vehicle_detections(self, result_dict):
        """筛选车辆检测结果（按类别名筛车辆）"""
        allowed_names = {"car", "truck", "bus", "motorcycle"}
        vehicle_detections = []

        cat_ids = result_dict.get('category_id', [])
        bboxes = result_dict.get('bbox', [])
        scores = result_dict.get('score', [])
        # print(f"检测结果总数: {len(cat_ids)}")
        if not cat_ids:
            print("警告: 没有检测到任何目标")
            return vehicle_detections

        detected_names = []
        for category_id, bbox, score in zip(cat_ids, bboxes, scores):
            # 将类别 id 转为 names 的索引
            if self.is_coco_dataset:
                if category_id not in COCO80_TO_COCO91_CLASS:
                    continue
                try:
                    class_idx = COCO80_TO_COCO91_CLASS.index(category_id)
                except ValueError:
                    continue
            else:
                class_idx = int(category_id)
                if class_idx < 0 or class_idx >= len(self.data_names):
                    continue
            class_name = self.data_names[class_idx]
            detected_names.append(class_name)

            if class_name in allowed_names:
                # 将 xywh 转 xyxy
                x, y, w, h = bbox
                xyxy_bbox = [x, y, x + w, y + h]
                vehicle_detections.append({
                    'bbox': xyxy_bbox,
                    'class_id': category_id,
                    'class_name': class_name,
                    'confidence': score
                })

        print(f"按名称筛选的车辆目标数: {len(vehicle_detections)}；名称分布: {set(detected_names)}")
        return vehicle_detections

    def _detect_suspicious_behavior(self, track_id: int, current_time: float):
        """检测可疑车辆行为"""
        if track_id not in self.tracked_vehicles:
            return
            
        vehicle = self.tracked_vehicles[track_id]
        suspicious_behaviors = []
        suspicious_score = 0.0
        
        # 1. 异常速度行为
        if vehicle.speed > 100.0:  # 超速
            suspicious_behaviors.append("Speeding!!!")
            suspicious_score += 2.0
        elif vehicle.speed < 20.0 and vehicle.lane_id != self.lane_detector.emergency_lane_id:
            suspicious_behaviors.append("Extremely Slow!!!")
            suspicious_score += 1.0
            
        # 2. 频繁变道
        if len(vehicle.positions_history) >= 5:
            # 检测是否频繁跨越车道边界
            lane_changes = 0
            prev_lane = None
            for i in range(1, len(vehicle.positions_history)):
                current_pos = vehicle.positions_history[i]
                # 简化的变道检测（基于x坐标变化）
                if prev_lane is not None:
                    x_change = abs(current_pos[0] - vehicle.positions_history[i-1][0])
                    if x_change > 50:  # 像素阈值
                        lane_changes += 1
                prev_lane = vehicle.lane_id
                
            if lane_changes > 3:  # 频繁变道
                suspicious_behaviors.append("Frequent Lane Changes!!!")
                suspicious_score += 1.5
                vehicle.last_lane_change = current_time
        
        # 3. 逆向行驶检测
        if len(vehicle.positions_history) >= 3:
            # 检测车辆是否朝错误方向移动
            start_y = vehicle.positions_history[0][1]
            end_y = vehicle.positions_history[-1][1]
            # 假设正常方向是从上到下（y增加）
            if end_y < start_y - 30:  # 向上移动超过30像素
                suspicious_behaviors.append("Suspected reverse Driving!!!")
                suspicious_score += 3.0
        
        # 4. 长时间停留在应急车道
        if (vehicle.lane_id == self.lane_detector.emergency_lane_id and 
            vehicle.speed < 10.0 and 
            (current_time - vehicle.entry_time) > 30.0):
            suspicious_behaviors.append("Long Stay in Emergency Lane!!!")
            suspicious_score += 2.5
        
        # 如果超过阈值，加入可疑车辆列表
        if suspicious_score >= self.suspicious_threshold:
            self.suspicious_vehicles[track_id] = vehicle
            # 更新可疑行为记录
            vehicle.suspicious_behaviors = suspicious_behaviors
            vehicle.suspicious_score = suspicious_score
            print(f"检测到可疑车辆 ID:{track_id}, 行为:{suspicious_behaviors}, 评分:{suspicious_score:.1f}")
    
    def _process_tracked_vehicles(self, tracked_detections, current_time: float):
        """处理跟踪车辆并更新统计数据"""
        # 重置车道车辆计数
        for lane_id in self.lane_detector.lanes:
            self.lane_detector.lanes[lane_id]['vehicle_count'] = 0
            # 初始化车道车辆分类统计
            if lane_id not in self.lane_vehicle_stats:
                self.lane_vehicle_stats[lane_id] = {
                    'car': 0, 'truck': 0, 'bus': 0, 'motorcycle': 0, 
                    'bicycle': 0, 'person': 0, 'unknown': 0
                }
            else:
                # 重置当前帧的统计
                for vehicle_type in self.lane_vehicle_stats[lane_id]:
                    self.lane_vehicle_stats[lane_id][vehicle_type] = 0
        
        # 重置全局车辆类别统计（当前帧）
        current_frame_stats = {
            'car': 0, 'truck': 0, 'bus': 0, 'motorcycle': 0, 
            'bicycle': 0, 'person': 0, 'unknown': 0
        }
        
        for detection in tracked_detections:
            track_id = detection['track_id']
            bbox = detection['bbox']
            # 优先使用 class_name，回退到 id 映射
            vehicle_type = detection.get('class_name')
            if not vehicle_type:
                class_id = detection.get('class_id')
                vehicle_type = self.vehicle_types.get(class_id, 'unknown')
            
            # 获取车辆中心位置
            center_x = int((bbox[0] + bbox[2]) / 2)
            center_y = int((bbox[1] + bbox[3]) / 2)
            current_position = (center_x, center_y)
            
            # 确定车道
            lane_id = self.lane_detector.get_vehicle_lane(bbox)
            if lane_id is not None:
                self.lane_detector.lanes[lane_id]['vehicle_count'] += 1
                # 更新车道车辆分类统计
                if vehicle_type in self.lane_vehicle_stats[lane_id]:
                    self.lane_vehicle_stats[lane_id][vehicle_type] += 1

            # 更新全局车辆分类统计
            if vehicle_type in current_frame_stats:
                current_frame_stats[vehicle_type] += 1

            # 更新累计统计（仅新车辆）
            if track_id not in self.tracked_vehicles:
                if vehicle_type in self.vehicle_count_stats:
                    self.vehicle_count_stats[vehicle_type] += 1
            
            # 更新或创建车辆信息
            if track_id not in self.tracked_vehicles:
                # 根据车道和位置生成真实速度
                estimated_speed = 50.0
                if lane_id is not None and self.flow_manager:
                    estimated_speed = self.flow_manager.estimate_realistic_speed(lane_id, center_y)
                self.tracked_vehicles[track_id] = VehicleInfo(
                    track_id=track_id,
                    vehicle_type=vehicle_type,
                    speed=estimated_speed, # 将根据位置历史计算
                    estimated_speed=estimated_speed,
                    lane_id=lane_id,
                    entry_time=current_time,
                    last_position=current_position,
                    positions_history=deque(maxlen=10),
                    speed_timestamp=current_time,
                    suspicious_behaviors=[],  
                    suspicious_score=0.0      
                )
            else:
                vehicle = self.tracked_vehicles[track_id]
                vehicle.vehicle_type = vehicle_type
                vehicle.last_position = current_position
                vehicle.positions_history.append(current_position)
                vehicle.lane_id = lane_id

                # 定期更新估计速度（每2秒一次）
                if current_time - vehicle.speed_timestamp > 2.0:
                    if lane_id is not None and self.flow_manager:
                        vehicle.estimated_speed = self.flow_manager.estimate_realistic_speed(lane_id, center_y)
                    vehicle.speed_timestamp = current_time
                
                # 根据位置历史计算速度
                if len(vehicle.positions_history) >= 2:
                    vehicle.speed = self._calculate_speed(vehicle.positions_history)
            
            # 更新车道统计数据
            if lane_id is not None:
                self.lane_detector.update_lane_stats(lane_id, self.tracked_vehicles[track_id].speed)

            # 交通流量分析流程
            if self.flow_manager:
                self.flow_manager.process_vehicle(track_id, current_position, lane_id, current_time)
            
            # 检查应急车道违规行为
            self._check_emergency_violations(track_id, lane_id, current_time)

            # 新增：检测可疑行为
            self._detect_suspicious_behavior(track_id, current_time)
        
        # 存储当前帧统计以供显示
        self.current_frame_vehicle_stats = current_frame_stats

    def _calculate_speed(self, positions: deque) -> float:
        """根据位置历史计算速度"""
        if len(positions) < 2:
            return 0.0
        
        # 计算总移动距离
        total_distance = 0
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            total_distance += np.sqrt(dx*dx + dy*dy)
        
        frames = len(positions) - 1
        if frames <= 0:
            return 0.0
        
        # 像素到速度转换
        # 假设: 每像素约0.15米，30FPS
        pixels_per_meter = 6.67  # 1米约6.67像素
        fps = 30
        
        # 速度计算: 距离(米) / 时间(秒) * 3.6 (转km/h)
        meters_per_second = (total_distance / pixels_per_meter) / (frames / fps)
        speed_kmh = meters_per_second * 3.6
        
        # 如果计算出的速度太低或太高，返回0表示无效
        if speed_kmh < 1.0 or speed_kmh > 150.0:
            return 0.0
        
        return round(speed_kmh, 1)
    
    def _check_emergency_violations(self, track_id: int, lane_id: int, current_time: float):
        """检查应急车道违规行为"""
        if (lane_id == self.lane_detector.emergency_lane_id and 
            track_id in self.tracked_vehicles):
            
            vehicle = self.tracked_vehicles[track_id]
            violation_duration = current_time - vehicle.entry_time

            # 等待足够的时间让速度计算稳定
            if violation_duration < 3.0:  # 至少等待3秒
                return
                
            # 使用估计速度作为备选
            display_speed = vehicle.speed if vehicle.speed > 5.0 else vehicle.estimated_speed
            
            # 优化应急车道检测 - 检查车辆是否行驶缓慢
            is_slow = vehicle.speed < 30.0
            
            if violation_duration > 2.0 and is_slow:  # 占用应急车道超过2秒且行驶缓慢则视为违规
                violation_info = {
                    'track_id': track_id,
                    'vehicle_type': vehicle.vehicle_type,
                    'speed': display_speed,
                    'duration': violation_duration,
                    'timestamp': current_time,
                    'lane_position': vehicle.last_position
                }
                
                # 如未存在则添加至违规记录
                if not any(v['track_id'] == track_id for v in self.emergency_violations):
                    self.emergency_violations.append(violation_info)
    
    def _draw_annotations(self, frame: np.ndarray, tracked_detections) -> np.ndarray:
        """在帧上绘制所有标注"""
        # 绘制当前时间
        current_datetime = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, current_datetime, (frame.shape[1] - 260, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 绘制处理时间
        proc_time = time.time() - self.start_time
        fps = self.frame_count / proc_time if proc_time > 0 else 0
        cv2.putText(frame, f"Time: {proc_time:.1f}s FPS: {fps:.1f}", (frame.shape[1] - 260, 60), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # 绘制车道多边形
        for lane_id, lane_info in self.lane_detector.lanes.items():
            # 根据拥堵情况用颜色绘制车道
            if lane_info['is_emergency']:
                color = (0, 0, 255)  # 紧急车道用红色标示
            elif lane_info['congestion_level'] == 'Heavy':
                color = (0, 0, 200)  # 深红色表示严重拥堵
            elif lane_info['congestion_level'] == 'Moderate':
                color = (0, 140, 255)  # 橙色代表温和
            else:
                color = (0, 255, 0)  # 绿色表示正常

            cv2.polylines(frame, [lane_info['polygon']], True, color, 2)
            
            # 车道信息文本
            centroid = np.mean(lane_info['polygon'], axis=0).astype(int)
            label = f"Lane {lane_id}"
            if lane_info['is_emergency']:
                label += " (EMERGENCY)"
            cv2.putText(frame, label, tuple(centroid), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
        # 绘制入口/出口区域
        if self.flow_manager:
            frame = self.flow_manager.draw_zones(frame)
        
        # 绘制车辆检测与追踪结果
        for detection in tracked_detections:
            track_id = detection['track_id']
            bbox = detection['bbox']
            bbox = [int(x) for x in bbox]
            
            # 根据车道确定颜色（应急车道以红色高亮显示）
            if track_id in self.tracked_vehicles:
                vehicle = self.tracked_vehicles[track_id]
                # 新增：可疑车辆用特殊颜色标记
                if track_id in self.suspicious_vehicles:
                    box_color = (0, 255, 255)  # 黄色表示可疑车辆
                    # 绘制可疑车辆警告标记
                    warning_pt = (bbox[0], bbox[1]-40)
                    cv2.circle(frame, warning_pt, 15, (0, 255, 255), -1)
                    cv2.putText(frame, "!", warning_pt, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
                elif vehicle.lane_id == self.lane_detector.emergency_lane_id:
                    box_color = (0, 0, 255)  # 红色为应急车道
                else:
                    box_color = (255, 0, 0)  # 蓝色表示普通车道

                if vehicle.lane_id == self.lane_detector.emergency_lane_id:
                    box_color = (0, 0, 255)  # 红色为应急车道
                    # 检查是否违规
                    is_violation = any(v['track_id'] == track_id for v in self.emergency_violations)
                    if is_violation:
                        box_color = (0, 0, 255)  # 红色代表违规
                        # 绘制警告三角形
                        warning_pt1 = (bbox[0]-10, bbox[1]-30)
                        warning_pt2 = (bbox[0]+10, bbox[1]-10)
                        warning_pt3 = (bbox[0]-30, bbox[1]-10)
                        warning_pts = np.array([warning_pt1, warning_pt2, warning_pt3], np.int32)
                        cv2.fillPoly(frame, [warning_pts], (0, 0, 255))
                else:
                    box_color = (255, 0, 0)  # 蓝色表示普通车道
            else:
                box_color = (255, 0, 0)
            
            # 绘制边界框
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), box_color, 2)
            
            # 车辆信息
            if track_id in self.tracked_vehicles:
                vehicle = self.tracked_vehicles[track_id]
                # 如果有实测速度则使用，否则使用估计值
                display_speed = vehicle.speed if vehicle.speed > 5.0 else vehicle.estimated_speed

                # info_text = f"ID:{track_id} {vehicle.vehicle_type} {display_speed:.1f}km/h"
                # cv2.putText(frame, info_text, (bbox[0], bbox[1]-10), 
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # 新增：显示可疑行为信息
                if track_id in self.suspicious_vehicles:
                    info_text = f"ID:{track_id} {vehicle.vehicle_type} {display_speed:.1f}km/h SUSPICIOUS"
                    # 显示可疑行为详情
                    for i, behavior in enumerate(vehicle.suspicious_behaviors[:2]):  # 最多显示2个行为
                        behavior_text = f"  {behavior}"
                        cv2.putText(frame, behavior_text, (bbox[0], bbox[1]-30+i*15), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                else:
                    info_text = f"ID:{track_id} {vehicle.vehicle_type} {display_speed:.1f}km/h"
                    
                cv2.putText(frame, info_text, (bbox[0], bbox[1]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 绘制流量统计图
        self._draw_traffic_stats(frame)

        # 绘制流量统计
        if self.flow_manager:
            frame = self.flow_manager.draw_flow_stats(frame, start_y=250)
        
        # 绘制紧急违规情况
        self._draw_emergency_violations(frame)

        # 绘制可疑车辆统计
        self._draw_suspicious_vehicles_stats(frame)
        
        return frame
    
    def _draw_vehicle_classification_stats(self, frame: np.ndarray):
        """绘制车辆分类统计信息"""
        # 绘制总车辆分类统计
        stats_start_y = 100
        cv2.putText(frame, "=== VEHICLE CLASSIFICATION ===", (10, stats_start_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        y_offset = stats_start_y + 25
        total_current = sum(self.current_frame_vehicle_stats.values())
        total_cumulative = sum(self.vehicle_count_stats.values())
        
        cv2.putText(frame, f"Current Frame Total: {total_current} | Cumulative Total: {total_cumulative}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 20
        
        # 显示各类车辆统计
        for vehicle_type in ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'person']:
            current_count = self.current_frame_vehicle_stats.get(vehicle_type, 0)
            total_count = self.vehicle_count_stats.get(vehicle_type, 0)
            
            if current_count > 0 or total_count > 0:
                color = (0, 255, 0) if current_count > 0 else (128, 128, 128)
                stats_text = f"{vehicle_type.capitalize()}: {current_count} ({total_count} total)"
                cv2.putText(frame, stats_text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y_offset += 18
        
        # 绘制按车道分类的车辆统计
        lane_stats_y = y_offset + 10
        cv2.putText(frame, "=== LANE VEHICLE DISTRIBUTION ===", (10, lane_stats_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        lane_stats_y += 25
        for lane_id in sorted(self.lane_vehicle_stats.keys()):
            lane_stats = self.lane_vehicle_stats[lane_id]
            lane_total = sum(lane_stats.values())
            
            if lane_total > 0:
                # 车道标题
                lane_color = (0, 0, 255) if self.lane_detector.lanes[lane_id].get('is_emergency', False) else (0, 255, 0)
                lane_title = f"Lane {lane_id}"
                if self.lane_detector.lanes[lane_id].get('is_emergency', False):
                    lane_title += " (EMERGENCY)"
                lane_title += f": {lane_total} vehicles"
                
                cv2.putText(frame, lane_title, (10, lane_stats_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, lane_color, 2)
                lane_stats_y += 15
                
                # 车道内车辆分类
                for vehicle_type, count in lane_stats.items():
                    if count > 0:
                        vehicle_text = f"  - {vehicle_type}: {count}"
                        cv2.putText(frame, vehicle_text, (20, lane_stats_y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                        lane_stats_y += 12

    def _draw_suspicious_vehicles_stats(self, frame: np.ndarray):
        """绘制可疑车辆统计信息"""
        if self.suspicious_vehicles:
            # 绘制可疑车辆标题
            suspicious_text = f"SUSPICIOUS VEHICLES: {len(self.suspicious_vehicles)}"
            cv2.rectangle(frame, (10, frame.shape[0]-150), 
                         (len(suspicious_text)*12, frame.shape[0]-120), (0, 255, 255), -1)
            cv2.putText(frame, suspicious_text, (20, frame.shape[0] - 130), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # 显示最近的可疑车辆信息
            y_offset = frame.shape[0] - 110
            for i, (track_id, vehicle) in enumerate(list(self.suspicious_vehicles.items())[-3:]):
                behaviors_str = ", ".join(vehicle.suspicious_behaviors[:2])  # 最多显示2个行为
                suspicious_info = f"ID:{track_id} {vehicle.vehicle_type} - {behaviors_str}"
                cv2.putText(frame, suspicious_info, (20, y_offset + i * 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 200), 1)

    def build_summary(self):
        """构建包含车辆分类统计的完整摘要"""
        end_time = time.time()
        
        # 车道统计
        lane_stats = []
        if self.lane_detector:
            for lane_id, info in self.lane_detector.lanes.items():
                lane_vehicle_breakdown = self.lane_vehicle_stats.get(lane_id, {})
                lane_stats.append({
                    "lane_id": lane_id,
                    "is_emergency": bool(info.get("is_emergency")),
                    "vehicle_count": int(info.get("vehicle_count", 0)),
                    "avg_speed": float(self.lane_detector.get_average_speed(lane_id)),
                    "congestion_level": info.get("congestion_level", "Unknown"),
                    "vehicle_breakdown": dict(lane_vehicle_breakdown)  # 车道内车辆分类
                })
        
        # 违规统计
        violations = []
        for v in self.emergency_violations:
            violations.append({
                "track_id": v["track_id"],
                "vehicle_type": v["vehicle_type"],
                "speed": float(v["speed"]),
                "duration": float(v["duration"]),
                "timestamp": v["timestamp"],
                "time_str": time.strftime("%H:%M:%S", time.localtime(v["timestamp"]))
            })
        
        # 可疑车辆统计
        suspicious_vehicles = []
        for track_id, vehicle in self.suspicious_vehicles.items():
            suspicious_vehicles.append({
                "track_id": track_id,
                "vehicle_type": vehicle.vehicle_type,
                "suspicious_behaviors": vehicle.suspicious_behaviors,
                "suspicious_score": float(vehicle.suspicious_score),
                "speed": float(vehicle.speed),
                "lane_id": vehicle.lane_id
            })
        
        # 构建完整摘要
        summary = {
            "datetime": time.strftime('%Y-%m-%d %H:%M:%S'),
            "total_time_sec": round(end_time - self.start_time, 2),
            "total_frames": self.frame_count,
            "fps_estimated": round(self.frame_count / (end_time - self.start_time), 2) if end_time > self.start_time else 0.0,
            
            # 新增：车辆分类统计
            "vehicle_classification": {
                "cumulative_counts": dict(self.vehicle_count_stats),
                "total_vehicles_detected": sum(self.vehicle_count_stats.values()),
                "current_frame_counts": dict(self.current_frame_vehicle_stats),
                "current_frame_total": sum(self.current_frame_vehicle_stats.values())
            },
            
            "lanes": lane_stats,
            "emergency_violations_count": len(violations),
            "emergency_violations": violations[:50],  # 截断防止过大
            "suspicious_vehicles_count": len(suspicious_vehicles),
            "suspicious_vehicles": suspicious_vehicles[:20]  # 限制数量
        }
        
        return summary
    
    def _draw_traffic_stats(self, frame: np.ndarray):
        """在框架上绘制流量统计"""
        y_offset = 30
        
        for lane_id, lane_info in self.lane_detector.lanes.items():
            avg_speed = self.lane_detector.get_average_speed(lane_id)
            stats_text = f"Lane {lane_id}: {lane_info['vehicle_count']} vehicles, "
            stats_text += f"Avg Speed: {avg_speed:.1f}km/h, {lane_info['congestion_level']}"
            
            color = (0, 255, 0)
            if lane_info['congestion_level'] == 'Heavy':
                color = (0, 0, 255)
            elif lane_info['congestion_level'] == 'Moderate':
                color = (0, 165, 255)
            
            cv2.putText(frame, stats_text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 25
    
    def _draw_emergency_violations(self, frame: np.ndarray):
        """绘制紧急车道违章警告"""
        if self.emergency_violations:
            # 绘制大号警告
            warning_text = f"EMERGENCY LANE VIOLATIONS: {len(self.emergency_violations)}"
            cv2.rectangle(frame, (10, frame.shape[0]-80), 
                         (len(warning_text)*13, frame.shape[0]-40), (0, 0, 100), -1)
            cv2.putText(frame, warning_text, (20, frame.shape[0] - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # 显示最近的违规行为及更多信息
            for i, violation in enumerate(self.emergency_violations[-3:]):
                violation_time = time.strftime("%H:%M:%S", time.localtime(violation['timestamp']))
                violation_text = f"Vehicle ID {violation['track_id']} - {violation['vehicle_type']} - {violation_time} - {violation['speed']:.1f}km/h"
                cv2.putText(frame, violation_text, (20, frame.shape[0] - 20 + i * 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 1)

def create_sample_config(config_path: str):
    """当文件不存在时才创建示例配置"""
    if os.path.exists(config_path):
        print(f"Lane configuration file already exists: {config_path}")
        return
    
    config = {
        "video_width": 1920,
        "video_height": 1080,
        "lanes": [
            {
                "id": 0,
                "points": [
                    [0, 150],         # 左下
                    [400, 0],            # 左上
                    [480, 0],          # 右上
                    [0, 250]        # 右下
                ],
                "is_emergency": True
            },
            {
                "id": 1,
                "points": [
                    [0, 250],       # 左下
                    [480, 0],          # 左上
                    [510, 0],          # 右上
                    [0, 400]        # 右下
                ],
                "is_emergency": False
            },
            {
                "id": 2,
                "points": [
                    [0, 400],       # 左下
                    [510, 0],          # 左上
                    [560, 0],         # 右上
                    [50, 750]        # 右下
                ],
                "is_emergency": False
            },
            {
                "id": 3,
                "points": [
                    [50, 750],       # 左下
                    [560, 0],         # 左上
                    [630, 0],         # 右上
                    [350, 1080]       # 右下
                ],
                "is_emergency": False
            },
            {
                "id": 4,
                "points": [
                    [350, 1080],       # 左下
                    [630, 0],         # 左上
                    [700, 0],         # 右上
                    [960, 1080]       # 右下
                ],
                "is_emergency": False
            },
            {
                "id": 5,
                "points": [
                    [960, 1080],       # 左下
                    [700, 0],         # 左上
                    [750, 0],         # 右上
                    [1500, 1080]       # 右下
                ],
                "is_emergency": False
            },
            {
                "id": 6,
                "points": [
                    [1500, 1080],       # 左下
                    [750, 0],         # 左上
                    [850, 0],         # 右上
                    [1920, 900]       # 右下
                ],
                "is_emergency": True
            }
        ]
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Sample configuration created: {config_path}")

def main():
    # 配置信息
    config_path = "./configs/yolov12/yolov12-s.yaml"  # YOLOv12n配置文件
    weight_path = "./EMA_yolov12-s-600_462.ckpt"  # 训练好的权重文件
    lane_config_path = "lane_config.json"
    video_path = "test_video.mp4"
    
    # 检查必要文件是否存在
    print("=== 检查文件存在性 ===")
    if not os.path.exists(config_path):
        print(f"错误: 配置文件不存在: {config_path}")
        print("请确保YOLOv11配置文件存在")
        return
    
    if not os.path.exists(video_path):
        print(f"错误: 视频文件不存在: {video_path}")
        print("请确保测试视频文件存在")
        return
    
    if not os.path.exists(weight_path):
        print(f"警告: 权重文件不存在: {weight_path}")
        print("模型将使用随机初始化，这会导致检测效果很差")
        print("建议下载预训练的YOLOv11权重文件")
    
    # 如果车道配置文件不存在，则创建
    create_sample_config(lane_config_path)
    
    # 初始化处理器
    processor = LaneVehicleProcessor(config_path, weight_path, lane_config_path)
    
    # 处理视频
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return
    
   # 获取视频信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"视频信息: 总帧数={total_frames}, FPS={fps}")

    # 初始化视频处理
    width, height = processor.initialize_video_processing(video_path)
    print(f"视频分辨率: {width}x{height}")
    
    # 保存输出视频的选项
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_lane_detection.mp4', fourcc, 30.0, (width, height))
    
    print("开始处理视频...")
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 处理框架
        annotated_frame = processor.process_frame(frame)
        
        # 将帧保存到输出视频
        out.write(annotated_frame)
        
        # 显示结果
        # cv2.imshow("MindSpore Lane Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    # cv2.destroyAllWindows()
    
    # 打印最终统计信息
    print("\n=== 车道检测分析总结 ===")
    print(f"日期时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总处理时间: {time.time() - processor.start_time:.2f} 秒")
    print(f"总处理帧数: {processor.frame_count}")

    # 车辆分类统计
    print("\n--- 车辆分类统计 ---")
    total_vehicles = sum(processor.vehicle_count_stats.values())
    print(f"累计检测到的车辆总数: {total_vehicles}")
    for vehicle_type, count in processor.vehicle_count_stats.items():
        if count > 0:
            percentage = (count / total_vehicles) * 100 if total_vehicles > 0 else 0
            print(f"  {vehicle_type.capitalize()}: {count} ({percentage:.1f}%)")
    
    print("\n--- 按车道分类的车辆分布 ---")
    for lane_id in sorted(processor.lane_vehicle_stats.keys()):
        lane_stats = processor.lane_vehicle_stats[lane_id]
        lane_total = sum(lane_stats.values())
        if lane_total > 0:
            lane_title = f"车道 {lane_id}"
            if processor.lane_detector.lanes[lane_id].get('is_emergency', False):
                lane_title += " (应急车道)"
            print(f"{lane_title}: 总计 {lane_total} 辆")
            for vehicle_type, count in lane_stats.items():
                if count > 0:
                    print(f"  - {vehicle_type}: {count} 辆")
    
    print("\n--- 车道统计 ---")
    for lane_id, lane_info in processor.lane_detector.lanes.items():
        avg_speed = processor.lane_detector.get_average_speed(lane_id)
        print(f"车道 {lane_id}: 平均速度: {avg_speed:.1f}km/h, "
            f"拥堵状况: {lane_info['congestion_level']}")
    
    print("\n--- 应急车道违规情况 ---")
    print(f"总违规次数: {len(processor.emergency_violations)}")
    for i, v in enumerate(processor.emergency_violations[:5]):
        violation_time = time.strftime("%H:%M:%S", time.localtime(v['timestamp']))
        print(f"{i+1}. 车辆 {v['track_id']} ({v['vehicle_type']}) 在 {violation_time}, "
            f"持续时间: {v['duration']:.1f}s, 速度: {v['speed']:.1f}km/h")

    # 可疑车辆统计
    if processor.suspicious_vehicles:
        print("\n--- 可疑车辆统计 ---")
        print(f"检测到可疑车辆: {len(processor.suspicious_vehicles)} 辆")
        for track_id, vehicle in list(processor.suspicious_vehicles.items())[:5]:
            behaviors = ", ".join(vehicle.suspicious_behaviors[:3])
            print(f"  车辆 {track_id} ({vehicle.vehicle_type}): {behaviors} (评分: {vehicle.suspicious_score:.1f})")
    
    print("\n输出视频已保存为: output_lane_detection.mp4")

def run_lane_detection(config_path: str, weight_path: str, lane_config_path: str, video_path: str, output_path: str = "output_lane_detection.mp4"):
    """Run lane detection pipeline with given params and save annotated video."""
    # 初始化处理器
    processor = LaneVehicleProcessor(config_path, weight_path, lane_config_path)

    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频文件: {video_path}")

    # 初始化
    width, height = processor.initialize_video_processing(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    out = cv2.VideoWriter(output_path, fourcc, float(fps), (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        annotated = processor.process_frame(frame)
        out.write(annotated)

    cap.release()
    out.release()
    # 生成统计概要
    summary = processor.build_summary()
    return output_path, summary

if __name__ == "__main__":
    main()