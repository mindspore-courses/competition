#!/usr/bin/env python3
# Author: YUYE
# Create Time: 2024/7/8

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
from mindformers import LlamaTokenizer

# 设置命令行参数解析器
parser = argparse.ArgumentParser(description="Read the first 1500 entries from task1, or the first 500 entries from task2.")
parser.add_argument("-T", "--task", help="Task number. 1 for alpaca_5010.json, 2 for alpaca_521.json", required=True, type=int)
args = parser.parse_args()

# 请确保cut是偶数
if   args.task == 1:
    cut = 1500
    data_path = "../../performance_serving/alpaca_5010.json"
elif args.task == 2:
    cut = 500
    data_path = "../../performance_serving/alpaca_521.json"
else:
    raise ValueError("Invalid task number")


# 定义读取数据集并分析token序列长度的函数
def analyze_data():
    # 定义初始化分词器
    tokenizer = LlamaTokenizer("../../performance_serving/tokenizer.model")

    # 读取数据集
    with open(data_path, "r", encoding="utf-8") as f:
        alpaca_data = json.load(f)[:cut]

    # 分析token序列长度
    input_token_lengths  = []
    output_token_lengths = []

    for data in alpaca_data:
        input_ = data["instruction"] + ":" + data["input"] if data["input"] else data["instruction"]
        input_token_lengths .append(len(tokenizer.tokenize(input_)))
        output_token_lengths.append(len(tokenizer.tokenize(data["output"])))

    return input_token_lengths, output_token_lengths

input_token_lengths, output_token_lengths = analyze_data()

# 定义绘制并保存直方图和折线图的函数
def plot_hist_and_line(input_token_lengths, output_token_lengths):
    # 绘制直方图
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(input_token_lengths, bins=50, color='blue', alpha=0.7)
    plt.title("Input Token Length Histogram")
    plt.xlabel("Token Length")
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    plt.hist(output_token_lengths, bins=50, color='green', alpha=0.7)
    plt.title("Output Token Length Histogram")
    plt.xlabel("Token Length")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(os.path.basename(data_path).replace(".json", f"_{cut}_Token_Length_Histogram.png"), dpi=300)

    # 绘制折线图
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sorted_input_lengths = sorted(input_token_lengths)
    plt.plot(sorted_input_lengths, color='blue')
    plt.title("Input Token Length Line Plot")
    plt.xlabel("Rank")
    plt.ylabel("Token Length")

    plt.subplot(1, 2, 2)
    sorted_output_lengths = sorted(output_token_lengths)
    plt.plot(sorted_output_lengths, color='green')
    plt.title("Output Token Length Line Plot")
    plt.xlabel("Rank")
    plt.ylabel("Token Length")

    plt.tight_layout()
    plt.savefig(os.path.basename(data_path).replace(".json", f"_{cut}_Token_Length_Line_Plot.png"), dpi=300)

plot_hist_and_line(input_token_lengths, output_token_lengths)

# 定义统计并保存数据的函数
def save_stats(input_token_lengths, output_token_lengths):
    stats = {
        "input_token_lengths": {
            "max": max(input_token_lengths),
            "min": min(input_token_lengths),
            "avg": np.array(input_token_lengths).mean(),
            "median": sorted(input_token_lengths)[len(input_token_lengths) // 2],
            "mode": max(set(input_token_lengths), key=input_token_lengths.count),
            "variance": np.array(input_token_lengths).var(),
            "std_dev": np.array(input_token_lengths).std(),
        },
        "output_token_lengths": {
            "max": max(output_token_lengths),
            "min": min(output_token_lengths),
            "avg": np.array(output_token_lengths).mean(),
            "median": sorted(output_token_lengths)[len(output_token_lengths) // 2],
            "mode": max(set(output_token_lengths), key=output_token_lengths.count),
            "variance": np.array(output_token_lengths).var(),
            "std_dev": np.array(output_token_lengths).std(),
        },
    }
    return stats

stats = save_stats(input_token_lengths, output_token_lengths)
stats_path = os.path.basename(data_path).replace(".json", f"_{cut}_stats.json")

with open(stats_path, "w", encoding="utf-8") as of:
    json.dump(stats, of, indent=4, ensure_ascii=False)
