import numpy as np

def get_path(env, model, test_case):
    
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
            'path': [test_case['attractions'][i] for i in info['path']],
            'total_distance': info['total_distance'],
            'total_cost': info['total_cost'],
            'total_score': info['total_score'],
            'budget_remaining': info['budget_remaining'],
            'attractions_visited': info['attractions_visited'],
            'total_reward': total_reward
        }

        
        print(f"最终路径: {result['path']}")
        print(f"访问景点数: {result['attractions_visited']}")
        print(f"总距离: {result['total_distance']:.2f}")
        print(f"总花费: {result['total_cost']:.2f}")
        print(f"景点平均评分: {result['total_score']/result['attractions_visited']:.2f}")
        # print(f"剩余预算: {result['budget_remaining']:.2f}")
        # print(f"总奖励: {result['total_reward']:.2f}")
        # print("-" * 50)
    
    return result