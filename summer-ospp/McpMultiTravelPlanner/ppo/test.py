import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from touristAttractions import TouristAttractionEnv

def generate_training_data(num_datasets=50):
    """生成训练数据集"""
    training_datasets = []
    
    # 常见景点名称库
    attraction_names = [
        "故宫", "天安门", "颐和园", "长城", "天坛", "北海公园", "恭王府", 
        "圆明园", "明十三陵", "雍和宫", "南锣鼓巷", "什刹海", "景山公园",
        "鸟巢", "水立方", "798艺术区", "国家博物馆", "王府井", "三里屯",
        "香山公园", "北京大学", "清华大学", "动物园", "海洋馆", "世界公园"
    ]
    
    for i in range(num_datasets):
        # 随机选择景点数量（10-20个）
        num_attractions = random.randint(10, 20)
        attractions = random.sample(attraction_names, num_attractions)
        
        # 生成距离矩阵（对称矩阵）
        distance_matrix = np.random.uniform(1, 100, (num_attractions, num_attractions))
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
        np.fill_diagonal(distance_matrix, 0)
        
        # 生成评分（3-5分）
        attraction_scores = np.random.uniform(3.0, 5.0, num_attractions)
        
        # 生成花费（20-150元）
        attraction_costs = np.random.uniform(20, 150, num_attractions)
        
        # 随机选择1-3个免费景点
        free_indices = random.sample(range(num_attractions), random.randint(1, 3))
        for idx in free_indices:
            attraction_costs[idx] = 0
        
        # 随机选择预算类型
        budget_types = ["economy", "comfort", "luxury"]
        user_budget_type = random.choice(budget_types)
        
        # 随机选择最小访问景点数（3-8个）
        min_visit = random.randint(3, 8)
        
        training_datasets.append({
            'attractions': attractions,
            'distance_matrix': distance_matrix,
            'attraction_scores': attraction_scores,
            'attraction_costs': attraction_costs,
            'user_budget_type': user_budget_type,
            'min_attractions': min_visit
        })
    
    return training_datasets

def generate_test_data():
    """生成测试数据集"""
    test_cases = []
    
    # 测试案例1：北京经典景点（12个）
    beijing_attractions = [
        "故宫", "天安门", "颐和园", "长城", "天坛", "北海公园", 
        "恭王府", "圆明园", "雍和宫", "南锣鼓巷", "鸟巢", "王府井"
    ]
    
    beijing_distances = np.array([
        [0, 1, 15, 70, 5, 2, 3, 20, 6, 4, 12, 3],
        [1, 0, 14, 69, 4, 1, 2, 19, 5, 3, 11, 2],
        [15, 14, 0, 55, 12, 13, 12, 5, 11, 13, 8, 14],
        [70, 69, 55, 0, 65, 68, 67, 60, 66, 68, 62, 69],
        [5, 4, 12, 65, 0, 3, 4, 17, 3, 5, 10, 4],
        [2, 1, 13, 68, 3, 0, 1, 18, 4, 2, 11, 1],
        [3, 2, 12, 67, 4, 1, 0, 17, 3, 1, 10, 2],
        [20, 19, 5, 60, 17, 18, 17, 0, 16, 18, 13, 19],
        [6, 5, 11, 66, 3, 4, 3, 16, 0, 4, 9, 5],
        [4, 3, 13, 68, 5, 2, 1, 18, 4, 0, 11, 1],
        [12, 11, 8, 62, 10, 11, 10, 13, 9, 11, 0, 12],
        [3, 2, 14, 69, 4, 1, 2, 19, 5, 1, 12, 0]
    ])
    
    beijing_scores = np.array([4.8, 4.5, 4.7, 4.9, 4.3, 4.2, 4.4, 4.1, 4.6, 4.0, 4.3, 4.2])
    beijing_costs = np.array([60, 0, 40, 80, 20, 15, 45, 25, 30, 0, 50, 0])
    
    test_cases.append({
        'name': '北京经典12景点',
        'attractions': beijing_attractions,
        'distance_matrix': beijing_distances,
        'attraction_scores': beijing_scores,
        'attraction_costs': beijing_costs,
        'user_budget_type': 'comfort',
        'min_attractions': 6
    })
    
    # 测试案例2：上海现代景点（15个）
    shanghai_attractions = [
        "外滩", "东方明珠", "南京路", "豫园", "田子坊", "上海博物馆",
        "静安寺", "迪士尼", "陆家嘴", "新天地", "世纪公园", "朱家角",
        "上海科技馆", "中华艺术宫", "七宝老街"
    ]
    
    shanghai_distances = np.random.uniform(5, 50, (15, 15))
    shanghai_distances = (shanghai_distances + shanghai_distances.T) / 2
    np.fill_diagonal(shanghai_distances, 0)
    
    shanghai_scores = np.random.uniform(3.8, 4.9, 15)
    shanghai_costs = np.random.uniform(0, 120, 15)
    shanghai_costs[[1, 3, 5, 9]] = 0  # 设置几个免费景点
    
    test_cases.append({
        'name': '上海现代15景点',
        'attractions': shanghai_attractions,
        'distance_matrix': shanghai_distances,
        'attraction_scores': shanghai_scores,
        'attraction_costs': shanghai_costs,
        'user_budget_type': 'luxury',
        'min_attractions': 8
    })
    
    # 测试案例3：杭州自然景点（18个）
    hangzhou_attractions = [
        "西湖", "雷峰塔", "灵隐寺", "宋城", "西溪湿地", "千岛湖",
        "龙井村", "六和塔", "岳王庙", "河坊街", "宝石山", "九溪烟树",
        "太子湾", "胡雪岩故居", "中国美院", "南宋御街", "白堤", "苏堤"
    ]
    
    hangzhou_distances = np.random.uniform(2, 40, (18, 18))
    hangzhou_distances = (hangzhou_distances + hangzhou_distances.T) / 2
    np.fill_diagonal(hangzhou_distances, 0)
    
    hangzhou_scores = np.random.uniform(4.0, 5.0, 18)
    hangzhou_costs = np.random.uniform(10, 80, 18)
    hangzhou_costs[[0, 9, 15]] = 0  # 设置免费景点
    
    test_cases.append({
        'name': '杭州自然18景点',
        'attractions': hangzhou_attractions,
        'distance_matrix': hangzhou_distances,
        'attraction_scores': hangzhou_scores,
        'attraction_costs': hangzhou_costs,
        'user_budget_type': 'economy',
        'min_attractions': 7
    })
    
    return test_cases

def train_model(env, training_datasets, total_timesteps=50000):
    """训练模型"""
    print("开始训练模型...")
    model = PPO("MlpPolicy", env, verbose=1, 
                learning_rate=0.0003, n_steps=2048, batch_size=64)
    
    # 使用多种景点数量进行训练
    for i, data in enumerate(training_datasets):
        print(f"训练数据集 {i+1}/{len(training_datasets)}: {len(data['attractions'])}个景点")
        
        env.set_attraction_data(
            attractions=data['attractions'],
            distance_matrix=data['distance_matrix'],
            attraction_scores=data['attraction_scores'],
            attraction_costs=data['attraction_costs'],
            user_budget_type=data['user_budget_type'],
            min_attractions=data['min_attractions']
        )
        
        # 每个数据集训练一定步数
        model.learn(total_timesteps=total_timesteps // len(training_datasets),
                   reset_num_timesteps=False)
    
    return model

def test_model(env, model, test_cases):
    """测试模型"""
    results = []
    
    for test_case in test_cases:
        print(f"\n=== 测试: {test_case['name']} ===")
        print(f"景点数量: {len(test_case['attractions'])}")
        print(f"预算类型: {test_case['user_budget_type']}")
        print(f"最大访问数: {test_case['min_attractions']}")
        
        env.set_attraction_data(
            attractions=test_case['attractions'],
            distance_matrix=test_case['distance_matrix'],
            attraction_scores=test_case['attraction_scores'],
            attraction_costs=test_case['attraction_costs'],
            user_budget_type=test_case['user_budget_type'],
            min_attractions=test_case['min_attractions']
        )
        
        obs, _ = env.reset()
        done = False
        total_reward = 0
        path = []
        step_count = 0
        max_steps = 100
        deterministic = True
        
        while not done and step_count < max_steps:
            # 获取当前有效动作
            valid_actions = env.get_valid_actions()
            
            # 使用模型预测
            action, _states = model.predict(obs, deterministic=deterministic)
            
            # 安全验证动作
            if action < 0 or action >= len(valid_actions):
                print(f"预测的动作 {action} 超出范围，选择终止动作")
                action = env.max_attractions
            elif not valid_actions[action]:
                # 从有效动作中选择
                valid_indices = np.where(valid_actions)[0]
                if len(valid_indices) > 0:
                    action = valid_indices[0]  # 选择第一个有效动作
                else:
                    action = env.max_attractions
            
            obs, reward, done, _, info = env.step(action)
            total_reward += reward
            path = info['path']
            step_count += 1
            
            if step_count >= max_steps:
                print(f"达到最大步数 {max_steps}，强制终止")
    
        
        # 记录结果
        result = {
            'name': test_case['name'],
            'path': [test_case['attractions'][i] for i in info['path']],
            'total_distance': info['total_distance'],
            'total_cost': info['total_cost'],
            'total_score': info['total_score'],
            'budget_remaining': info['budget_remaining'],
            'attractions_visited': info['attractions_visited'],
            'total_reward': total_reward
        }
        
        results.append(result)
        
        print(f"最终路径: {result['path']}")
        print(f"访问景点数: {result['attractions_visited']}")
        print(f"总距离: {result['total_distance']:.2f}")
        print(f"总花费: {result['total_cost']:.2f}")
        print(f"总评分: {result['total_score']:.2f}")
        print(f"剩余预算: {result['budget_remaining']:.2f}")
        print(f"总奖励: {result['total_reward']:.2f}")
        print("-" * 50)
    
    return results



if __name__ == "__main__":
    # 初始化环境
    env = TouristAttractionEnv(max_attractions=25)
    
    # 模型文件路径
    model_path = "./flexible_tourist_model_10_20_min"  # 修改为你的模型路径
    
    # 加载模型
    print("正在加载模型...")
    model = PPO.load(model_path, env=env)
    print("模型加载成功！")
    
    # 生成测试数据
    print("\n生成测试数据...")
    test_cases = generate_test_data()
    
    # 测试模型
    print("\n开始测试模型...")
    results = test_model(env, model, test_cases)
    
    # 输出测试结果摘要
    print("\n=== 测试结果摘要 ===")
    for result in results:
        print(f"{result['name']}:")
        print(f"  访问了 {result['attractions_visited']} 个景点")
        print(f"  总评分: {result['total_score']:.2f}, 总花费: {result['total_cost']:.2f}")
        print(f"  路径: {result['path']}")
        print() 