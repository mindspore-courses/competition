# 作品报告

## 1.	微调算法介绍

### 1.2	微调数据集规模的预处理
采用文档中fastchat prompt,抽取9w条数据，采用本地大模型对模型进行难度评分，设置1-5分难度，选取1，2分难度数据8w条，3，4，5难度数据1w条进行训练。

## 2.	超参配置介绍说明

```
learning_rate: 5.e-6
pet_config:
      pet_type: lora
      # configuration of lora
      lora_rank: 16
      lora_alpha: 32
      lora_dropout: 0.05
      target_modules: '.*wq|.*wv'
      
```
调小学习率以减少原有能力遗忘，由于加大秩，原有能力遗忘增加，但是学的更多，选择r=16

## 3.	微调后权重文件链接
https://owl.obs.cn-southwest-2.myhuaweicloud.com/new_lora_checkpoint_0.ckpt
https://owl.obs.cn-southwest-2.myhuaweicloud.com/new_lora_checkpoint_1.ckpt
https://owl.obs.cn-southwest-2.myhuaweicloud.com/new_lora_checkpoint_2.ckpt
https://owl.obs.cn-southwest-2.myhuaweicloud.com/new_lora_checkpoint_3.ckpt
https://owl.obs.cn-southwest-2.myhuaweicloud.com/new_lora_checkpoint_4.ckpt

## 4.	运行环境说明
按指导书进行环境配置

## 5.	模型微调后原有能力评估得分
F1 score: 57.75580434860321, Em score: 43.54136429608128, total_count: 2067

## 6.	推理方式修改

为了保持训练prompt和推理prompt一致，在推理脚本中加入prompt，`“Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{problem}\n\n### Response:”`
