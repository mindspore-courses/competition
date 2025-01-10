#!/usr/bin/env python3
# Author: YUYE
# Create Time: 2024/7/4

import json
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import os

# 设置命令行参数解析器
parser = ArgumentParser()
parser.add_argument("-I", "--in_fp", type=str, required=True, help="Input JSON file path")
parser.add_argument("-O", "--out_fp", type=str, help="Output PNG file path (default: same as input file with .png extension)")
args = parser.parse_args()

# 读取JSON文件
with open(args.in_fp, 'r', encoding='utf-8') as file:
    data = json.load(file)

# 如果没有提供输出文件路径，使用输入文件名替换扩展名为.png
if args.out_fp is None:
    args.out_fp = os.path.splitext(args.in_fp)[0] + '.png'

# 提取res_time和first_token_time
res_times = [entry['res_time'] for entry in data]
first_token_times = [entry['first_token_time'] for entry in data]
num_samples = len(res_times)

# 创建折线图
plt.figure(figsize=(15, 5))  # 设置图表大小

# 绘制res_time折线
plt.plot(res_times, label='res_time', linestyle='-', color='blue')

# 绘制first_token_time折线
plt.plot(first_token_times, label='first_token_time', linestyle='--', color='red')

# 设置图表标题和图例
plt.title('Response Time and First Token Time')
plt.xlabel('Sample Number')
plt.ylabel('Time')
plt.legend()

# 根据样本数量设置横坐标的显示方式
if num_samples <= 100:
    plt.xticks(range(0, num_samples + 1, 5), range(0, num_samples + 1, 5))
elif num_samples <= 200:
    plt.xticks(range(0, num_samples + 1, 10), range(0, num_samples + 1, 10))
elif num_samples <= 500:
    plt.xticks(range(0, num_samples + 1, 50), range(0, num_samples + 1, 50))
else:
    plt.xticks(range(0, num_samples + 1, 100), range(0, num_samples + 1, 100))


# 保存图表到指定的输出文件
plt.savefig(args.out_fp, dpi=300)
