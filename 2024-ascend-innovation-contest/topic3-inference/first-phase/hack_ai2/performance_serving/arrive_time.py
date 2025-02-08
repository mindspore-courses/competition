import numpy as np

def generate_arrival_time(rate, cv, num_jobs):
    scale = 1 / rate
    shape = cv**-2
    inter_arrival_times = np.random.gamma(shape, scale, num_jobs)
    arrival_times = np.cumsum(inter_arrival_times)
    return arrival_times.tolist()

# 设置到达率和变异系数
arrival_rate = 0.5
cv = 0.2
num_jobs = 100

# 生成到达时间
arrival_times = generate_arrival_time(arrival_rate, cv, num_jobs)
print(arrival_times)
