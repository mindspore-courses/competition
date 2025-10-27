from typing import List
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

class TouristAttractionEnv(gym.Env):
    """改进的景点路径规划环境，考虑评分、花费和用户预算类型; 支持不同景点数量"""
    
    def __init__(self, max_attractions: int = 20):
        super(TouristAttractionEnv, self).__init__()
        
        self.max_attractions = max_attractions
        
        # 固定大小的动作空间
        self.action_space = spaces.Discrete(max_attractions + 1)
        
        # 固定大小的状态空间
        self.observation_space = spaces.Box(
            low=0, 
            high=100, 
            shape=(max_attractions * 3 + 7,),
            dtype=np.float32
        )
        
        # 将在reset时设置的实际数据
        self.attractions = None
        self.n_attractions = 0
        self.distance_matrix = None
        self.attraction_scores = None
        self.attraction_costs = None
        self.user_budget_type = "comfort"
        self.daily_budget = 0
        self.min_attractions = 5
        self.cost_reward = 0
    
    def set_attraction_data(self, 
                          attractions: List[str],
                          distance_matrix: np.ndarray,
                          attraction_scores: np.ndarray,
                          attraction_costs: np.ndarray,
                          user_budget_type: str = "comfort",
                          daily_budget: float = None,
                          min_attractions: int = None):
        """设置当前行程的景点数据"""
        self.n_attractions = len(attractions)
        if self.n_attractions > self.max_attractions:
            raise ValueError(f"景点数量({self.n_attractions})超过最大限制({self.max_attractions})")
        
        self.attractions = attractions
        self.distance_matrix = distance_matrix.astype(np.float32)
        self.attraction_scores = attraction_scores.astype(np.float32)
        self.attraction_costs = attraction_costs.astype(np.float32)
        self.user_budget_type = user_budget_type
        self.min_attractions = min_attractions if min_attractions is not None else min(5, self.n_attractions)
        
        if self.min_attractions > self.n_attractions:
            raise ValueError(f"最少访问景点数({self.min_attractions})不能超过总景点数({self.n_attractions})")
    
    def _get_default_budget(self) -> float:
        """根据预算类型获取默认预算"""
        if self.user_budget_type == "economy":
            return np.mean(self.attraction_costs) * self.min_attractions * 1.2
        elif self.user_budget_type == "comfort":
            return np.mean(self.attraction_costs) * self.min_attractions * 1.5
        else:  # luxury
            return np.mean(self.attraction_costs) * self.min_attractions * 2.0
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.attractions is None:
            raise ValueError("请先调用set_attraction_data设置景点数据")
        
        # 初始化状态变量
        self.visited = np.zeros(self.max_attractions, dtype=np.float32)
        self.current_pos = -1
        self.path = []
        self.total_distance = 0.0
        self.total_cost = 0.0
        self.total_score = 0.0
        self.steps_taken = 0
        self.has_started = False
        self.budget_remaining = self._get_default_budget()
        self.min_attractions_met = False
        self.cost_reward = 0
        self.invalid_action_count = 0  # 跟踪无效动作次数
        
        return self._get_obs(), {}
    
    def _get_obs(self):
        """获取观察值，处理可变景点数量"""
        # 当前位置的one-hot编码
        current_pos_onehot = np.zeros(self.max_attractions, dtype=np.float32)
        if 0 <= self.current_pos < self.n_attractions:
            current_pos_onehot[self.current_pos] = 1
        
        # 每个景点的成本与剩余预算的比例
        cost_ratio = np.zeros(self.max_attractions, dtype=np.float32)
        for i in range(self.n_attractions):
            if self.budget_remaining > 0:
                cost_ratio[i] = self.attraction_costs[i] / self.budget_remaining
            else:
                cost_ratio[i] = 10.0
        
        # 对于超出实际景点数量的位置，设置掩码值
        for i in range(self.n_attractions, self.max_attractions):
            cost_ratio[i] = -1.0
        
        obs = np.concatenate([
            self.visited,
            current_pos_onehot,
            cost_ratio,
            np.array([self.steps_taken], dtype=np.float32),
            np.array([max(0, self.min_attractions - self.steps_taken)], dtype=np.float32),
            np.array([self.total_distance], dtype=np.float32),
            np.array([self.total_cost], dtype=np.float32),
            np.array([self.budget_remaining], dtype=np.float32),
            np.array([1.0 if self.has_started else 0.0], dtype=np.float32),
            np.array([1.0 if self.min_attractions_met else 0.0], dtype=np.float32)
        ])
        return obs
    
    def _get_valid_actions(self):
        """获取有效动作掩码"""
        valid_actions = np.ones(self.max_attractions + 1, dtype=bool)
        
        # 标记超出实际景点数量的动作为无效
        for i in range(self.n_attractions, self.max_attractions):
            valid_actions[i] = False
        
        # 标记已访问的景点为无效
        for i in range(self.n_attractions):
            if self.visited[i] == 1:
                valid_actions[i] = False
        
        # 只有达到最少访问要求后才能选择终止动作
        if self.steps_taken < self.min_attractions:
            valid_actions[self.max_attractions] = False
        elif self.steps_taken >= self.n_attractions:  # 所有景点都访问过了
            valid_actions[self.max_attractions] = True
            # 将所有景点动作标记为无效，只能选择终止
            for i in range(self.n_attractions):
                valid_actions[i] = False
        
        return valid_actions
    
    def get_valid_actions(self):
        """公开方法获取有效动作掩码"""
        return self._get_valid_actions()
    
    def step(self, action):
        terminated = False
        truncated = False
        reward = 0
        
        # 获取有效动作掩码
        valid_actions = self._get_valid_actions()
        
        # 关键修复：确保 action 在有效范围内
        if action < 0 or action >= len(valid_actions) or action >= self.n_attractions:
            reward = -1000
            terminated = True
            print(f"动作 {action} 超出有效范围 [0, {self.n_attractions-1}]")
        
        # 无效动作处理
        elif not valid_actions[action]:
            self.invalid_action_count += 1
            reward = -1000
            terminated = True
            print(f"无效动作: {action}, 有效动作: {np.where(valid_actions)[0]}")
        
        # 终止动作
        elif action == self.max_attractions:
            if self.steps_taken < self.min_attractions:
                reward = -500 * (self.min_attractions - self.steps_taken)
            else:
                # 最终奖励计算
                reward = (self.steps_taken * 10 + 
                         self.total_score * 6 - 
                         self.total_distance * 0.5 - 
                         self.cost_reward * 0.2)
                if self.steps_taken > self.min_attractions:
                    reward += (self.steps_taken - self.min_attractions) * 8
            terminated = True
        
        # 有效景点动作
        else:
            print(f"有效动作：{action}")
            # 预算检查
            cost_penalty = 0
            if self.attraction_costs[action] > self.budget_remaining:
                cost_penalty = 2 * (self.attraction_costs[action] - self.budget_remaining)
                self.cost_reward += cost_penalty
            
            # 计算移动成本
            if not self.has_started:
                distance_cost = 0
                start_bonus = 15  # 开始奖励
                self.has_started = True
            else:
                distance_cost = self.distance_matrix[self.current_pos, action]
                start_bonus = 0
            
            # 更新状态
            self.total_distance += distance_cost
            self.total_cost += self.attraction_costs[action]
            self.total_score += self.attraction_scores[action]
            self.budget_remaining -= self.attraction_costs[action]
            self.visited[action] = 1
            self.current_pos = action
            self.path.append(action)
            self.steps_taken += 1
            
            # 基础奖励计算
            reward = (start_bonus + 
                    8 +  # 访问基础奖励
                    self.attraction_scores[action] * 3 -  # 评分奖励
                    distance_cost * 0.3 -  # 距离惩罚
                    cost_penalty * 0.1)  # 成本惩罚
            
            # 达到最少访问景点数的奖励
            if not self.min_attractions_met and self.steps_taken >= self.min_attractions:
                self.min_attractions_met = True
                reward += 200
            
            # 探索奖励
            unvisited_count = np.sum(self.visited[:self.n_attractions] == 0)
            reward += unvisited_count * 0.3
            
            # 检查是否所有景点都已访问
            if self.steps_taken >= self.n_attractions:
                terminated = True
                reward += 100  # 完成所有景点的额外奖励
        
        info = {
            'path': self.path.copy(),
            'total_distance': self.total_distance,
            'total_cost': self.total_cost,
            'total_score': self.total_score,
            'budget_remaining': self.budget_remaining,
            'attractions_visited': self.steps_taken,
            'min_attractions_met': self.min_attractions_met,
            'valid_actions': valid_actions,
            'invalid_action_count': self.invalid_action_count
        }
        
        return self._get_obs(), reward, terminated, truncated, info
    
    def render(self):
        print(f"当前路径: {[self.attractions[i] for i in self.path if i < self.n_attractions]}")
        print(f"已访问景点数: {self.steps_taken} (最少需访问: {self.min_attractions})")
        print(f"总距离: {self.total_distance:.2f}")
        print(f"总花费: {self.total_cost:.2f}")
        print(f"总评分: {self.total_score:.2f}")
        print(f"剩余预算: {self.budget_remaining:.2f}")
        print(f"是否达到最少访问要求: {'是' if self.min_attractions_met else '否'}")
        print(f"有效景点: {[i for i in range(self.n_attractions) if self.visited[i] == 0]}")
        print(f"当前位置: {self.current_pos}")