# ChatGLM3-32K

## 模型描述

ChatGLM3 是智谱AI和清华大学 KEG 实验室联合发布的新一代对话预训练模型。ChatGLM3-6B 是 ChatGLM3 系列中的开源模型，在保留了前两代模型对话流畅、部署门槛低等众多优秀特性的基础上，ChatGLM3-6B 引入了如下特性：更强大的基础模型，更完整的功能支持，更全面的开源序列

ChatGLM3-6B-32K在ChatGLM3-6B的基础上进一步强化了对于长文本的理解能力，能够更好的处理最多32K长度的上下文。

```text
@article{zeng2022glm,
  title={Glm-130b: An open bilingual pre-trained model},
  author={Zeng, Aohan and Liu, Xiao and Du, Zhengxiao and Wang, Zihan and Lai, Hanyu and Ding, Ming and Yang, Zhuoyi and Xu, Yifan and Zheng, Wendi and Xia, Xiao and others},
  journal={arXiv preprint arXiv:2210.02414},
  year={2022}
}
```

## 仓库介绍

`chatGLM3-6B-32K` 基于 `mindformers` 实现，主要涉及的文件有：

1. 模型具体实现：`mindformers/models/glm2`
   glm32k复用glm2的代码实现

    ```text
    glm2
        ├── __init__.py
        ├── glm2.py                  # 模型实现
        ├── glm2_config.py           # 模型配置项
        ├── glm2_modules.py          # 模组实现
        ├── glm2_tokenizer.py        # tokenizer
        └── glm2_transformer.py      # transformer层实现
    ```

2. 模型配置：`research/glm32k`

    ```bash
    glm32k
        └── finetune_glm32k.yaml           # Atlas 800T A2最佳性能全量微调启动配置
        └── predict_glm.yaml           # Atlas 800T A2推理配置
    ```

3. 数据处理脚本和任务启动脚本：`research/glm32k`

    ```bash
    glm32k
        └── glm32k_preprocess.py           # glm32k微调的数据前处理脚本
    ```

## 前期准备

### 环境要求

**MindFormers安装**以及**软硬件配套关系**参考[MindFormers安装](../../README.md#二MindFormers安装)和[版本匹配关系](../../README.md#三版本匹配关系)。

### 模型权重下载与转换(mindformers权重或huggingface权重选择使用即可)

#### mindformers权重直接使用

本仓库提供已经转换完成的预训练权重用于微调/推理，用户可自行从下方链接拉取后直接使用。

下载链接：

权重：https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/glm32k/glm32k.ckpt

词表：https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/glm32k/tokenizer.model

linux可用如下命令下载。

```shell
#!/bin/bash
mkdir -p ckpt/rank_0
cd ./ckpt/rank_0
wget https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/glm32k/glm32k.ckpt
wget https://huggingface.co/THUDM/chatglm3-6b-32k/tree/main/tokenizer.model
cd ../..
```

#### 从huggingface下载原始权重后转换

需要将整个工程下载下来。

[chatglm3-6b-32k](https://huggingface.co/THUDM/chatglm3-6b-32k)

如果使用git命令下载，下载前请先确保已安装git-lfs。

```shell
git lfs install
git clone https://huggingface.co/THUDM/chatglm3-6b-32k
```

```shell
#!/bin/bash
pip install torch==1.11.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install transformers==4.30.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

下载完成后，运行`mindformers/models/glm2/convert_weight.py`转换脚本，将huggingface的权重转换为完整的ckpt权重。

```shell
#!/bin/bash
cd {mindformers根目录}
python mindformers/models/glm2/convert_weight.py --torch_path TORCH_CKPT_DIR --mindspore_path MS_CKPT_NAME
```

```text
# 参数说明
TORCH_CKPT_DIR: huggingface权重保存目录路径
MS_CKPT_NAME: mindspore格式的权重保存文件名，如'saved_dir/glm32k.ckpt'
```

**注**: 请安装torch=2.1.1和transformers=4.33.0版本。

### 模型权重切分与合并

从huggingface或官方github仓库转换而来的权重通常是单卡权重，基于该权重进行多卡微调，评测，推理，涉及ckpt从单机策略到分布式策略的切换。

通常训练采用分布式训练，基于该权重进行评测，推理多采用单卡，涉及ckpt从分布式策略到单机策略的切换。

以上涉及到ckpt的单卡，多卡转换，详细教程请参考特性文档[模型权重切分与合并](../../docs/feature_cards/Transform_Ckpt.md)

## ChatGLM3-6B-32K

### 微调性能

| config                               | task              | Datasets  | SeqLength | metric | phase             | score     | performance(tokens/s/p) |
|--------------------------------------|-------------------|-----------|-----------| ------ | ----------------- | --------- |-------------------------|
| [ChatGLM3-6B-32K](./finetune_glm32k.yaml) | text_generation   | longbench | 32768     | -      | [finetune](#微调)  | -         | 777.91                  |

### 微调

#### 数据集准备-SFT微调数据集

当前提供[LongBench](https://huggingface.co/datasets/THUDM/LongBench/tree/main)长序列数据集的预处理和微调样例，用于对ChatGLM3-6B-32K模型进行微调。

LongBench数据集样式

```text
{
    "input": "任务的输入/指令，通常较短，比如QA中的问题、Few-shot任务中的提问等",
    "context": "任务所需的长语境文本，比如文档、跨文件代码、Few-shot任务中的few-shot样本",
    "answers": "由所有标准答案组成的列表",
    "length": "前三项文本的总长度（中、英文分别用字、词数统计）",
    "dataset": "本条数据所属数据集名称",
    "language": "本条数据的语言",
    "all_classes": "分类任务中的所有类别，非分类任务则为null",
    "_id": "每条数据的随机id"
}
```

#### 数据集处理

将LongBench数据集格式转换为AdGen数据集格式，以便复用mindformers的ADGenDataLoader来转换为微调使用的数据样式。启动命令：

```shell
cd research/glm32k
python glm32k_preprocess.py \
--data_path INPUT_DATA_PATH \
--output_path OUTPUT_PATH \
--prompt_config_file PROMPT_PATH
```

```text
# 参数说明
INPUT_DATA_PATH: 原始longbench数据所处的文件夹路径
OUTPUT_PATH：转换格式后的数据存储路径
PROMPT_PATH：longbench中不同数据对应的prompt
```

**注意**：

- Longbench数据集链接：[Longbench](https://huggingface.co/datasets/THUDM/LongBench/blob/main/data.zip)
- prompt_config_file链接：[prompt_config_file](https://github.com/THUDM/LongBench/blob/main/config/dataset2prompt.json)
- 具体Longbench数据集介绍请参见[官网](https://github.com/THUDM/LongBench)地址

#### 全参微调

全参微调需要多卡启动，以`LongBench`数据集为例，给出了默认配置文件`research/glm32k/finetune_glm32k.yaml`。

当前模型已支持使用**Flash Attention算法**进行全参微调，请参考 [Flash Attention使用文档](../../docs/feature_cards/Training_Algorithms.md#flash-attention)

- step 1. 修改`research/glm32k/finetune_glm32k.yaml`中相关配置

```text
output_dir: './output' # path to save checkpoint/strategy
load_checkpoint: './output/transformed_checkpoint/'          # 添加预训练权重路径
auto_trans_ckpt: False
only_save_strategy: False
resume_training: False
use_parallel: True
run_mode: 'finetune'
# dataset
train_dataset: &train_dataset
  data_loader:
    type: ADGenDataLoader
    dataset_dir: "/path/to/AdvertiseGen/train.json"
    shuffle: True
    phase: "train"
    version: 3
    origin_columns: ["content", "summary"]
  tokenizer:
    type: ChatGLM3Tokenizer
    vocab_file: "/path/to/tokenizer.model"                   # 添加字典文件
  max_source_length: 30720                                   # 长序列源数据长度
  max_target_length: 2047                                    # 长序列目标数据长度
```

**注意**：长序列模型的训练，max_source_length和max_target_length数值较大，需要根据实际业务数据设置对应数值

- step 2. 启动微调任务，按照以下步骤启动：

-[x] 1: 根据服务器节点数等信息，修改相应的配置。

```shell
# 以glm-6b-32k模型为例，默认配置单机8卡，如果节点数有变，需要修改相应的配置。
# 配置文件路径：./research/glm32k/finetune_glm32k.yaml
parallel_config:
  data_parallel: 1
  model_parallel: 2
  pipeline_stage: 4
  micro_batch_num: 16
  vocab_emb_dp: True
  gradient_aggregation_group: 4
```

-[x] 2: 执行运行脚本。

```shell
cd {mindformers根目录}
bash scripts/msrun_launcher.sh "run_mindformer.py --config research/finetune_glm32k.yaml --run_mode finetune"
```

```text
# 参数说明
config: 配置文件路径
run_mode: 运行模式，微调时设置为finetune
```

### 推理

大模型推理升级训推一体架构，实现脚本、分布式策略和运行时的统一，通过融合大算子降低推理时延，有效提升网络吞吐量。
在启动前，请先行在配置文件predict_glm.yaml中将processor.tokenizer.vocab_file的路径配置为实际路径；增量推理开关在配置文件中model.model_config.use_past位置；

注：推理当前仅支持8k长度

```yaml
processor:
  return_tensors: ms
  tokenizer:
    ...
    vocab_file: '/path/tokenizer.model'  # 修改为实际路径
    ...
model:
  model_config:
    ...
    use_past: True
    is_dynamic: True
    ...
```

相关文件的下载链接如下：[tokenizer.model](https://huggingface.co/THUDM/chatglm3-6b-32k/blob/main/tokenizer.model)

#### 基于generate的推理

以下为基于model.generate接口的自定义推理脚本，glm32k当前仅支持单卡推理。

```python
# predict_custom.py 文件
import os
import argparse

import numpy as np
import mindspore as ms
from mindformers import MindFormerConfig, ChatGLM2Config, ChatGLM3Tokenizer, TransformerOpParallelConfig, ChatGLM2ForConditionalGeneration
from mindformers import init_context
from mindformers.tools.utils import str2bool


def main(args):
    """main function."""
    # 多batch输入
    inputs = ["晚上睡不着应该怎么办", "使用python编写快速排序代码"]

    # set model config
    config = MindFormerConfig(args.yaml_file)

    # 初始化环境
    init_context(use_parallel=False,
                 context_config=config.context,
                 parallel_config=config.parallel)

    model_config = ChatGLM2Config(**config.model.model_config)
    model_config.batch_size = len(inputs)
    model_config.seq_length = args.seq_length
    if args.checkpoint_path:
        model_config.checkpoint_name_or_path = args.checkpoint_path
    print(f"config is: {model_config}")

    # build tokenizer
    tokenizer = ChatGLM3Tokenizer(args.tokenizer_path)
    # build model from config
    model = ChatGLM2ForConditionalGeneration(model_config)

    if isinstance(inputs, list):
        inputs_ids = tokenizer.build_batch_input(inputs)["input_ids"]
    else:
        inputs_ids = tokenizer.build_chat_input(inputs)["input_ids"]
    outputs = model.generate(inputs_ids,
                             max_length=model_config.max_decode_length,
                             do_sample=model_config.do_sample,
                             top_k=model_config.top_k,
                             top_p=model_config.top_p)
    for output in outputs:
        print(tokenizer.decode(output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer_path', default='/path/to/tokenizer.model', type=str,
                        help='tokenizer model path.')
    parser.add_argument('--checkpoint_path', default='', type=str,
                        help='set checkpoint path.')
    parser.add_argument('--yaml_file', default="", type=str,
                        help='predict yaml path')
    parser.add_argument('--seq_length', default=512, type=int,
                        help='predict max length')
    args = parser.parse_args()
    main(args)

# [gMASK]sop<|user|>
# 晚上睡不着应该怎么办<|assistant|>
# 晚上睡不着,可以参考下述建议:
# 1. 建立规律的睡眠时间表:每天在相同的时间上床和起床,有助于身体建立规律的睡眠时间表,更容易入睡。
# 2. 创造舒适的睡眠环境:确保睡眠环境舒适,安静,黑暗,凉爽,有助于入睡。
# 3. 避免刺激性物质:避免饮用咖啡因和酒精等刺激性物质,这些物质会影响睡眠。
# 4. 放松身心:在睡前放松身心,例如泡个热水澡,听些轻柔的音乐,读本书等,有助于入睡。
# 5. 避免使用电子设备:在睡前避免使用电子设备,例如手机,平板电脑等,这些设备发出的蓝光会抑制睡眠激素的分泌,影响睡眠。
# 6. 锻炼身体:适度的锻炼身体有助于睡眠,但避免在睡前进行剧烈运动。
# 7. 寻求专业帮助:如果长期存在睡眠问题,建议寻求专业医生的帮助。
#
# 如果以上建议都无法解决问题,建议咨询医生,了解更具体的解决方案。
# [gMASK]sop<|user|>
# 使用python编写快速排序代码<|assistant|>
# 快速排序（Quick Sort）是一种高效的排序算法，其基本思想是通过一趟排序将待排记录分隔成独立的两部分，其中一部分记录的关键字均比另一部分关键字小，然后分别对这两部分记录继续进行排序，以达到整个序列有序。
#
# 下面是使用 Python 编写的快速排序代码：
#
# ```python
# def quick_sort(arr):
#     if len(arr) <= 1:
#         return arr
#     pivot = arr[len(arr) // 2]
#     left = [x for x in arr if x < pivot]
#     middle = [x for x in arr if x == pivot]
#     right = [x for x in arr if x > pivot]
#     return quick_sort(left) + middle + quick_sort(right)
#
# arr = [3,6,8,10,1,2,1]
# print("原始数组：", arr)
# print("排序后的数组：", quick_sort(arr))
# ```
#
# 这段代码首先定义了一个名为 `quick_sort` 的函数，该函数接受一个列表作为参数。然后，我们选择列表中间的元素作为基准值（pivot），并将列表中的元素分为三部分：小于基准值的元素（left）、等于基准值的元素（middle）和大于基准值的元素（right）。最后，我们递归地对左右两部分进行快速排序，并将排序后的结果合并在一起。
#
# 运行这段代码，输出结果如下：
#
# ```
# 原始数组： [3, 6, 8, 10, 1, 2, 1]
# 排序后的数组： [1, 1, 2, 3, 6, 8, 10]
# ```
#
# 这就是使用 Python 编写的快速排序代码。
```

```text
# 参数说明
tokenizer_path: tokenizer.model路径
checkpoint_path: 权重路径
yaml_file: yaml配置路径
seq_length: 模型的seq_length
```
