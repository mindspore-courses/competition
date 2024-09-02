# 昇腾AI-模型微调大赛

## 微调算法介绍

LoRA，即Low-Rank Adaptation，它通过将权重矩阵分解成低秩矩阵的乘积，降低了参数数目，进而达到减少硬件资源、加速微调进程的目的。

LoRA在保留基座模型全部参数的同时，拆分出权重矩阵的更新并进行矩阵分解，通过调整训练这个由低秩矩阵乘积表示的更新矩阵来减少存储空间的同时保留了模型的质量和微调速度。

## 超参配置说明

训练过程的使用的yaml文件中的参数配置，其他没有变化：

```
runner_config:
  epochs: 5
  batch_size: 64
  sink_mode: True
  sink_size: 2
  
pet_config:
      pet_type: lora
      # configuration of lora
      lora_rank: 8
      lora_alpha: 16
      lora_dropout: 0.0
      target_modules: '.*wq|.*wv'
```

其中epochs为训练的总共epoch数目，batch_size为训练的批次大小

lora_rank表示低秩分解中矩阵的秩，target_modules表示使用lora的层，这里对query和value的映射层进行分解

lora_alpha为缩放参数，用于缩放生成的W。

## 微调后的权重链接

[下载链接](https://llmft.obs.cn-southwest-2.myhuaweicloud.com/new_lora_checkpoint_v1.ckpt)

## Mindformers下载链接
[下载链接](https://llmft.obs.cn-southwest-2.myhuaweicloud.com/mindformers.zip)
## 运行环境

- Mindspore==2.3.0RC2
- MindFormers
- tiktoken

## 模型微调后原有能力评估得分

原有能力评估得分为：

- F1-score：61.0657
- EM-score：46.4441


