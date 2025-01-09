# 模型微调大赛报告
## 微调算法介绍
### 🌅 数据方面

zero_shot 添加zero微调数据（给output添加Let's think step by step）


## 超参配置介绍
### 🍰 训练方面
1. 随机种子设置为 42
2. epoch 为10，batch_size 为2，8卡训练，时长约为9小时左右
3. learning_rate 由 5.e-5 改为 2.e-4
4. seq_length 由 256 改为了 512
5. lora_rank 由 8 改为了 16
6. lora_dropout 由 0.1 改为了 0.05
7. target_modules 由 '.*wq|.*wk|.*wv|.*wo'改为了 '.*wq|.*wo' 

参与微调的参数量

Network Parameters: 4194304

总参数量为 7321000000

低参比例 = 4194304 / 7321000000 = 0.000572914082775577

训练yaml 具体参见压缩包中的 finetune_internlm_7b_lora_mmlu_64G.yaml


## 运行环境说明
🌅 无额外配置

## 模型微调后原有能力评估得分

F1 score: 47.20811977048287, Em score: 27.285921625544265

超过模型原有能力

原有能力推理的yaml做了调整，具体参见压缩包中的

predict_internlm_7b_eval_squad.yaml

## 推理预测准确率（随机20000条准确率99.30%）
# 使用列表推导式来比较两个列表的元素
```
result = [option1 == option2 for option1, option2 in zip(out_list,check_list)]
def calculate_accuracy(results):
    correct_count = sum(results)
    total_count = len(results)
    accuracy = correct_count / total_count if total_count > 0 else 0
    return accuracy
# 计算准确率
accuracy = calculate_accuracy(result)
print(f"Accuracy: {accuracy * 100:.2f}%")  # 输出准确率
```

OBS：
https://competition.obs.cn-north-305.tjaicc.com/wfp-finetune/output/checkpoint0/rank_0/new_lora_check_point_internlm00.ckpt

https://competition.obs.cn-north-305.tjaicc.com/wfp-finetune/output/checkpoint1/rank_0/new_lora_check_point_internlm10.ckpt

https://competition.obs.cn-north-305.tjaicc.com/wfp-finetune/output/checkpoint2/rank_0/new_lora_check_point_internlm20.ckpt

https://competition.obs.cn-north-305.tjaicc.com/wfp-finetune/output/checkpoint3/rank_0/new_lora_check_point_internlm30.ckpt

https://competition.obs.cn-north-305.tjaicc.com/wfp-finetune/new_lora_check_point_internlm_best.ckpt


https://competition.obs.cn-north-305.tjaicc.com/wfp-finetune/output_best/
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










