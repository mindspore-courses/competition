# 参数微调赛事报告


## 介绍


昇腾AI创新⼤赛是⾯向AI开发者打造的顶级赛事，旨在⿎励产业开发者基于昇腾AI技术和产品，探索模型算

法、算⼦、加速库等融合创新和性能优化，加速AI与⾏业融合，促进开发者能⼒提升。


昇思MindSpore模型开发挑战赛作为昇腾AI创新⼤赛系列赛事之⼀，旨在培养昇思MindSpore和昇腾AI云服务

开发者，⿎励开发者基于昇思MindSpore和昇腾AI云服务进⾏模型&算法创新与实践，并丰富国内模式⽣态。

在本次⽐赛中，我们创建、提取、转换了训练数据集、基础指标评估测试集，配置了LoRA微调环境，对LLM：

llama3-8b模型进⾏了LoRA微调。旨在保持其原有能⼒的情况下，提升其对于数学问题的回答准确率。


## LoRA

本次我们使⽤LoRA作为我们的低参微调⽅法。

LoRA（Low-Rank Adaptation）是⼀种⽤于微调⼤型语⾔模型（LLM）的技术。其主要原理是通过添加 少量的低秩（low-rank）矩阵来微调模型的权重，从⽽在保持模型性能的同时显著减少训练参数和计算 资源。


### LoRA的原理


1. 低秩矩阵：传统的微调⽅法需要更新整个模型的权重，这对于⼤型模型来说⾮常耗时且计算资源需求⾼。 ⽽LoRA的核⼼思想是，只需在模型中添加低秩矩阵，以近似描述微调过程中权重的变化。低秩矩阵具有较 少的参数，因此计算成本低。
2. 权重更新：在LoRA中，模型的权重更新矩阵 ΔW 被分解为两个⼩矩阵 A 和 B ，使得 ( W' = W + ΔW , ΔW = A × B )，W'是训练更新后的矩阵。通过只更新这两个低秩矩阵 A 和 B ，⽽不是整
个权重矩阵 W ，可以实现对模型的微调。这样的需要的算⼒更少。

3. 优势：

    - 参数效率：LoRA⼤幅减少了需要更新的参数数量，从⽽降低了存储和计算成本。

    - 可扩展性：由于参数数量减少，LoRA可以更容易地应⽤于更⼤规模的模型和更⼤规模的数据集。
    - 性能保持：尽管更新的参数减少，LoRA在许多任务上仍能保持与传统微调⽅法相当的性能。


### LoRA的应⽤

1. ⾃然语⾔处理（NLP）：LoRA在许多NLP任务中，如⽂本⽣成、翻译和问答系统中都得到了⼴泛应⽤。

通过微调预训练模型，LoRA可以有效提升模型在特定任务上的表现。


2. 计算机视觉（CV）：LoRA同样可以应⽤于计算机视觉领域，例如图像分类和⽬标检测，通过微调预训练 的视觉模型来提⾼任务性能。

3. 跨领域应⽤：由于其⾼效的参数更新机制，LoRA还可以在跨领域任务中应⽤，例如多模态学习和跨语⾔迁 移学习。


## 微调数据集的规模预处理⽅式

指导⼿册中原本提供了约80w的数据集，对于转换后的conversation版本数据集，我们：

- 去除了提示词“Below is an instruction that describes a task. Write a response that appropriately completes the request.”以及“### Instruction:”等；
- 前80万条（中⽂输⼊）⾥随机采样88800条，后9993条（英⽂输⼊）⾥随机采样1200条；

- 采样过程中，剔除含乘法“...*...=...”、以及除法“.../...=...”的提问。

以下为训练集样本示例(展示json格式的⼀例）：


![输入图片说明](image/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-08-22%20172252.png)



## 超参配置说明

![输入图片说明](image/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-08-22%20172513.png)


选取5epochs，64batch_size(最⼤可能得批⼤⼩，128会导致⽆法微调），都是基于时间考量，可以将 时间压缩到6h每次training


## Mindformer代码包url

https://btbdzzz.obs.cn-southwest-2.myhuaweicloud.com/7_28_Success_Compressed


## 微调后权重⽂件链接与对应原有能⼒评估

依据训练得到的时间从早到晚依次：

1. https://btbdzzz.obs.cn-southwest-2.myhuaweicloud.com/ckpt_merged_res1

2. https://btbdzzz.obs.cn-southwest-2.myhuaweicloud.com/ckpt_merged_res2

3. https://btbdzzz.obs.cn-southwest-2.myhuaweicloud.com/ckpt_merged_res3

4. https://btbdzzz.obs.cn-southwest-2.myhuaweicloud.com/lora_ckpt_0727.ckpt

5. https://btbdzzz.obs.cn-southwest-2.myhuaweicloud.com/ckpt_7_28_merged


请注意，在我们最终的⾃⾏测量中，数学能⼒表现最好的是第⼆个权重。（acc=0.619)

后续应当发⽣了⼀定的过拟合问题，虽然不严重，但是并没有使得训练产⽣正向效果。

尽管原有能⼒仍有压榨空间，但是在后续训练中我们发现原有能⼒和⽬标数学能⼒均下降，继续训练被 认为不再有意义。


## 运⾏环境说明

⽆特殊，下⽅为常规配置
```
pip install mindspore==2.3.0RC2 2
export PYTHONPATH="${PYTHONPATH}:/home/ma-user/work/mindformers/" 4
pip install tiktoken
```

## Prompt说明

详⻅ run_llama3_test.py 中。添加了：

●      前prompt： Please answer the following math problems:\n

●      后prompt： \n###answer

![输入图片说明](image/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-08-22%20172633.png)


请注意，必须严格按照我们的prompt修改模型输⼊进⾏ predict 任务，或者直接使⽤我们提供的 `run_llama3_test.py` ⽂件进⾏predict任务。

除此之外，对于模型输⼊与推理不再有任何改动。

## 参数量


低参⽐例：3407872/8030000000=0.000424



