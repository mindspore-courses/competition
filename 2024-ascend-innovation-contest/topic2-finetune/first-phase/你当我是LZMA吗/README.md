# 微调实验说明

### 数据集准备

- 从 11 类问题中忽略困难样本后
  - 算数乘法
  - 次方
  - 开根
  - 分数化简
  - 已知自变量求函数值
- 均匀分布随机抽取 7500 个样本作为训练集 [data_easy_7500.json](./data_easy_7500.json)
- 使用 CoT_v2 进行文本预处理，详细定义见 [make_dataset.py](./make_dataset.py) 中的 `make_CoT` 函数
- 使用模板 `Below is an instruction that describes a grade school math problem. Write a response that gives the correct answer.\n\n### Instruction:\n{problem}\n\n### Response:` 作为引导 prompt


### 模型微调训练

ℹ 详细的配置文件: [run_llama3_8b_8k_800T_A2_64G_lora_dis_256_single.yaml](./run_llama3_8b_8k_800T_A2_64G_lora_dis_256_single.yaml)

```
[训练配置]
device: npu*1 (just singe card!) 
epoch: 2
bs: 4
lr: 3e-5
target_modules: .*wq|.*wv
n_param: 3407872  (微调参数量)
```

ℹ 微调后的模型权重: https://vhaktyr.obs.cn-southwest-2.myhuaweicloud.com/output/


### 模型评估

原有能力评估见日志 [logs/test_eval_finetune.log](./logs/test_eval_finetune.log)

```
F1 score: 71.18689963833116
Em score: 54.0396710208031
```

数学能力评估使用自测数据集 [data_200_random.json](./data_200_random.json)，见日志 [logs/test_eval_finetune_math.log](./logs/test_eval_finetune_math.log)，输出保存为 [logs/text_generation_result.txt](./logs/text_generation_result.txt) 和 [logs/result_npy.npy](./logs/result_npy.npy)；仅供参考 :)


----

### 如何使用我们的代码

- 参考 [init.sh](./init.sh) 构建运行环境
- 参考 [run_train.sh](./run_train.sh) 制作数据集 & 运行训练
- 参考 [run_eval.sh](./run_eval.sh) 运行评估 & 推理
