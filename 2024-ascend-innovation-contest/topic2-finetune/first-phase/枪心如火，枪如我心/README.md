# 模型微调大赛报告
## 微调算法介绍
### 🌅 数据方面

1. 总数据量为809993，其中有重复数据，去除重复数据后为548942，有的数据，例如“将
分数 5/6 进行简化。”出现了480次，会影响训练效果。

2. 原始数据量太大，训练时间长且会对模型的原有能力造成较大影响，因此不太适合用大量
数据直接进行训练，我采样了其中部分数据（共4w）进行训练，具体分布如下：

a. 解方程数据：9525

b. 加法数据：9525

c. 减法数据：9525

d. 乘法数据：0

e. 除法数据：9525

f. 应用题相关数据：

i. 打折问题：374

ii. 求平均值：300

iii. 计算销售额：300

iv. 求矩形面积：374

v. 计算0次方：10

vi. 计算1次方：10

vii. 计算2次方：10

viii. 计算平方根：424

ix. 求物体质量：90

x. 简化分数：36

3. 求平均值问题和计算销售额问题由Qwen2-7B生成了推理过程，除法运算、打折、平方根
计算、面积计算、质量计算由自己设计了推理过程

4. 对解方程和四则运算的问题使用qwen72b生成模板，缓解模型的对话能力下降问题

5. 对解方程、除法运算、打折、计算平方根结果保留一位小数

具体数据处理代码参考压缩包中的 split_math_data_blind.py


模型结构没有进行改动


## 超参配置介绍
### 🍰 训练方面
1. 随机种子设置为 42
2. epoch 为1，batch_size 为4，单卡训练，时长约为两小时左右
3. seq_length 由 256 改为了 512
4. lora_rank 由 8 改为了 16
5. lora_alpha 由 16 改为了 32
6. target_modules 由 '.*wq|.*wv' 改为了 '.*wq|.*wk|.*wv|.*wo'

参与微调的参数量

Network Parameters: 13631488

总参数量为 8030000000

低参比例 = 13631488 / 8030000000 = 0.001697570112079701

训练yaml 具体参见压缩包中的 run_llama3_8b_8k_800T_A2_64G_lora_dis_256.yaml


## 运行环境说明
🌅 无额外配置

## 模型微调后原有能力评估得分

F1 score: 63.81453421560693, Em score: 46.250604741170775

超过模型原有能力

原有能力推理的yaml做了调整，具体参见压缩包中的

run_llama3_8b_8k_800T_A2_64G_lora_256_base_eval.yaml

## 数学计算结果推理

🥇 将 run_llama3_test.py 中原本的

`predict_data.append(pro_list)`

改为了

`predict_data.append(f'Below is an instruction that describes a task. Write a response
that appropriately completes the request.\n\n### Instruction:\n{pro_list}\n\n###
Response:')`

和训练时的prompt格式一致

数学能力推理的yaml做了调整，具体参见压缩包中的

run_llama3_8b_8k_800T_A2_64G_lora_256_eval.yaml

## 其他材料

🏝 • 数据处理脚本：参见压缩包中的 split_math_data_blind.py

• 模型微调的完整日志：参见压缩包中的 log 文件夹和 msrun_log 文件夹

• yaml格式的配置文件：

    ◦ 训练：参见压缩包中的 run_llama3_8b_8k_800T_A2_64G_lora_dis_256.yaml

    ◦ 原有能力推理：参见压缩包中的run_llama3_8b_8k_800T_A2_64G_lora_256_base_eval.yaml

    ◦ 数学能力推理：参见压缩包中的run_llama3_8b_8k_800T_A2_64G_lora_256_eval.yaml

• mindformers源码包：参见压缩包中的 mindformers_wz.zip

• 原有能力评估的完整日志文件：参见压缩包中的 base_eval.log

• 更改后的数学计算结果推理脚本：参见压缩包中的 run_llama3_test.py










