# BERT（base-uncased）

## 模型介绍

BERT（Bidirectional Encoder Representations from Transformers）是一种基于Transformer架构的深度学习语言模型，由Google于2018年提出。**BERT（base-uncased）**是BERT模型的一个具体版本。

### 1. **模型结构**

- **Transformer层数**：12层
- **隐藏层维度**：768
- **自注意力头数**：12
- **参数总量**：约110M（1.1亿）
- **词表大小**：30,000

### 2. **应用场景**

BERT（base-uncased）广泛应用于以下自然语言处理任务：

- **文本分类**：情感分析、主题分类等。
- **命名实体识别 (NER)**：识别文本中的特定实体，如人名、地名等。
- **问答系统**：根据给定上下文回答问题。
- **句子匹配**：如自然语言推理（NLI）或语义相似性计算。

## 比赛规则

- 在精度保持不变的情况下，进行性能比拼，单token推理时间短者胜出。
- 精度保持不变：无法达到官方提供baseline/或模型精度降低的成绩无效，官方提供精度测试的UT；
- 单token推理时间：测试验证1000个token推理的平均时间（不包含prefill和decode的首token）；

## 优化方法

### 1. 使用 `mint`算子对 `ops`算子进行替换：

```
ms.ops -> ms.mint
```

### 2. 在BertModel中对BertEncoder进行静态图编译：

具体操作为，对Bert的encoder类写一个函数，并且打上jit，用于静态图编译。但是需要注意原本的encoder中前向传播的输出需要修改，否则无法通过静态图编译。更多详细请参考 `modeling_bert.py`文件。

```python
@mindspore.jit(jit_config=mindspore.JitConfig(jit_syntax_level='STRICT'))
def encoder_jit(
            encoder,
            embedding_output,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_values,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
        ):
    hidden_states, layer_outputs_tuple = encoder(
            embedding_output,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_values,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
        )
    return hidden_states, layer_outputs_tuple
```

## 优化结果

单token推理时间（秒）：`0.011345896515641007`

---

# CLIP

## 模型介绍

CLIP（Contrastive Language-Image Pre-training）是OpenAI提出的一种多模态模型，能够将文本和图像映射到同一个特征空间中，并通过对比学习的方式关联两种模态。CLIP的核心思想是通过大量的图文对（如从互联网收集的图像和对应的描述）进行预训练，让模型学习到图像和文本之间的语义关系。

CLIP的结构包括两个独立的编码器：

1. **文本编码器**：基于Transformer架构，用于处理输入文本，将其映射为向量表示。
2. **图像编码器**：通常采用ResNet或Vision Transformer（ViT）架构，用于提取图像特征。

通过对比学习，CLIP使得图像与其对应的文本描述的特征向量距离更近，而不相关的图文对距离更远。这种特性使得CLIP在许多任务中表现出强大的零样本能力，例如：

- 图像分类：无需额外训练，直接通过自然语言描述分类标签实现分类。
- 图像检索：根据文字描述找到匹配的图像。
- 文本生成：与其他生成模型结合，用于生成与图像相关的文字描述。

CLIP的多模态特性和零样本能力，使其在计算机视觉和自然语言处理的交叉领域具有重要意义。

## 比赛规则

- 在精度保持不变的情况下，进行性能比拼，单token推理时间短者胜出。
- 精度保持不变：无法达到官方提供baseline/或模型精度降低的成绩无效，官方提供精度测试的UT；
- 单token推理时间：测试验证1000个token推理的平均时间（不包含prefill和decode的首token）；

## 优化方法

### 1. 使用 `mint`算子对 `ops`算子进行替换：

```
ms.ops -> ms.mint
```

### 2. 对CLIP的Encoder进行静态图编译：

对clip的encoder类写一个函数，并且打上jit，用于静态图编译。但是需要注意原本的encoder（vision encoder & text encoder）中前向传播的输出需要修改，否则无法通过静态图编译。更多详细请参考 `modeling_clip.py`文件。

```python
@mindspore.jit(jit_config=mindspore.JitConfig(jit_syntax_level='STRICT'))
def encoder_jit(encoder, 
            inputs_embeds,
            attention_mask,
            causal_attention_mask,
            output_attentions,
            output_hidden_states,
            return_dict):
    return  encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            )
```

## 优化结果

单token推理时间（秒）：`0.022965397562708583	`
