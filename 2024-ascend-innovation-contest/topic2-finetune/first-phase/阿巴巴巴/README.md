

# 一、报告

## 1.微调算法介绍

使用lora算法进行微调，微调数据集规模为9万条，数据集来源为原始数据集随机挑选。

### 1.1微调的参数修改

```bash
runner_config:
  epochs: 2
  batch_size: 64

pet_config:
      pet_type: lora
      # configuration of lora
      lora_rank: 8
      lora_alpha: 16
      lora_dropout: 0.05
      target_modules: '.*wq|.*wv|.*wo'
```

### 1.2数据集预处理

修改了data_converter.py, 在问题前添加了prompt
```bash
You are an expert in mathematics, accurately calculating the following problem.
```
修改位置为
```python
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),

    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\nYou are an expert in mathematics, accurately calculating the following problem. {problem}\n\n### Response:"
    ),

}
```


数据格式如下：

```bash
 {
    "id": "1",
    "conversations": [
      {
        "from": "human",
        "value": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nYou are an expert in mathematics, accurately calculating the following problem. 计算 -6797.66 - 280.27 等于多少？\n\n### Response:"
      },
      {
        "from": "gpt",
        "value": "-6797.66 - 280.27 = -7077.93"
      }
    ]
  },
```


## 2.超参配置介绍说明



## 3.微调后的权重链接

https://abbb-finetune.obs.cn-southwest-2.myhuaweicloud.com/new_0.ckpt


## 4.运行环境说明

无改变

## 5.模型微调后原有能力评估得分

低参比例 = 5505024/8030000000

F1 score: 58.58684049098678, Em score: 43.154329946782774, total_count: 2067

## 6.测试集的推理结果


# 二、相关文件


1.提供模型微调的完整日志、yaml格式的配置文件；
https://abbb-finetune.obs.cn-southwest-2.myhuaweicloud.com/finetune.zip

2.提供能保障从数据预处理到模型推理全流程跑通的mindformers源码包
https://abbb-finetune.obs.cn-southwest-2.myhuaweicloud.com/mindformers.zip

3.原有能力评估的完整日志文件。
https://abbb-finetune.obs.cn-southwest-2.myhuaweicloud.com/eval.zip

