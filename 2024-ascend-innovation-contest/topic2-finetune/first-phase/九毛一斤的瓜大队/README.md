## 一、作品报告

1. 微调算法介绍，包含使用的微调数据集规模的预处理方式

微调算法介绍：lora微调

LoRA（Low-Rank Adaptation）微调算法是一种用于调整大型预训练模型的高效微调技术。它主要通过在模型的权重矩阵中引入低秩矩阵来实现微调，这些低秩矩阵的秩远小于原始权重矩阵的秩，从而显著减少了需要训练的参数数量。以下是对LoRA微调算法的详细介绍：

基本原理：LoRA微调算法的核心思想是在冻结预训练模型权重的基础上，通过注入可训练的低秩分解矩阵来近似全参数微调的效果。具体来说，它将原始模型的增量参数矩阵（即微调过程中学习到的参数变化）表示为两个参数量更小的矩阵的乘积，从而实现低秩近似。

实现过程：

①冻结预训练模型权重：在LoRA微调过程中，预训练模型的权重被冻结，不接受梯度更新。

②注入可训练的低秩分解矩阵：在模型的每一层中，注入两个可训练的矩阵A和B，其中A的输入维度和B的输出维度分别与原始模型的输入输出维度相同，而A的输出维度和B的输入维度是一个远小于原始模型输入输出维度的值。这两个矩阵的乘积（BA）用于近似原始模型的增量参数矩阵。

③训练低秩分解矩阵：在训练过程中，只更新矩阵A和B的参数，而预训练模型的权重保持不变。这样，通过训练少量的参数即可实现模型的微调。

④推理时合并权重：在推理时，将训练好的低秩分解矩阵（BA）与原始模型的权重合并，得到新的权重矩阵用于推理。由于推理时不需要额外的计算开销，因此LoRA微调算法在推理时与原始模型保持一致。

微调数据集的规模：从80万条中英文混合题目中筛选出10000条题目来进行微调。预处理的主要是：①对80万条中英文混合题目按照关键词进行分类，结果可分成10类。②然后随机从这10类题目中各自抽取1000条样例，并进行一些正则化操作，规范数学计算，最后组合成一个新的训练数据集。具体的预处理代码见`mindformers/wly_all_file/data_converter.py`。

2. 超参配置介绍说明

LoRA的主要配置如下，全部配置见run_llama3_8b_8k_800T_A2_64G_lora_dis_ 256.yaml附件：

```
    pet_config:
      pet_type: lora
      # configuration of lora
      lora_rank: 8
      lora_alpha: 16
      lora_dropout: 0.05
      target_modules: '.*wq|.*wv'

```

3. 微调后的权重文件

已加载到obs桶中，链接为：
https://npu-wly-llama3-8b-ckpt.obs.cn-southwest-2.myhuaweicloud.com/wly-new-llama3/new_lora_checkpoint_0.ckpt


4. 运行环境说明

与该赛题的配置环境一致，与“指导手册中5.2 环境配置中提及的操作”一致。

配置代码如下：
```
# 重启服务器后：
pip install mindspore==2.3.0RC2
cd mindformers/ 
bash build.sh
export PYTHONPATH="${PYTHONPATH}:/home/ma-user/work/mindformers/"
pip install tiktoken

```

5. 模型微调后原有能力评估得分及低参比例

F1 score: 69.1504769264647, Em score: 52.73343009192066, total_count: 2067

6. 作品验收时的数学计算推理方式

与原指导手册的要求一致，使用的推理程序也一致，即使用run_llama3_test.py即可。


## 二、模型微调的完整日志、yaml格式的配置文件

已提供在附件中。

## 三、数据预处理到模型推理全流程跑通的mindformers源码包

已提供在附件中，也可以通过obs下载：
https://npu-wly-llama3-8b-ckpt.obs.cn-southwest-2.myhuaweicloud.com/wly-new-llama3/mindformers.zip



## 四、原有能力评估的完整日志文件 
已提供在附件中。


## 五、数据预处理到模型推理的全流程(linux命令)

第一步：启动服务器，下载对应mindspore的版本，然后下载本队伍提供的mindformer，该包已经含有（均在/mindformers/wly_all_file目录下）：

①模型微调的yaml文件

②80万条中英文混合数学运算的训练数据集train.json

③原有能力的评估数据集dev-v1.1.json

⑥对原80万条数据集进行预处理的data_converter.py文件

整体命令如下：

```
# 重启服务器后：
pip install mindspore==2.3.0RC2

wget https://npu-wly-llama3-8b-ckpt.obs.cn-southwest-2.myhuaweicloud.com/mindformers.zip

cd mindformers/ 
bash build.sh

export PYTHONPATH="${PYTHONPATH}:/home/ma-user/work/mindformers/"
pip install tiktoken
```


第二步：进行指导手册的“5.3 模型权重和 tokenizer 文件准备”

整体命令如下：
```
# 先进入对应目录
cd /home/ma-user/work/

# 权重文件下载
wget https://2024-ascend-innovation-contest-mindspore.obs.cn-southwest-2.myhuaweicloud.com/topic2-finetune/llama3-8B.ckpt

# tokenizer.model文件的下载
wget https://2024-ascend-innovation-contest-mindspore.obs.cn-southwest-2.myhuaweicloud.com/topic2-finetune/tokenizer.model
```


第三步：数据预处理并转换成 MindRecord 格式

整体命令如下：

```
# 数据预处理
python /home/ma-user/work/mindformers/wly_all_file/data_converter.py \
--data_path /home/ma-user/work/mindformers/wly_all_file/train.json \
--output_path /home/ma-user/work/mindformers/wly_all_file/train-data-10000-conversation.json

# 将预处理后的数据集转换为 MindRecord 格式
python /home/ma-user/work/mindformers/research/llama3/llama_preprocess.py \
--dataset_type qa \
--input_glob /home/ma-user/work/mindformers/wly_all_file/train-data-10000-conversation.json \
--model_file /home/ma-user/work/tokenizer.model \
--seq_length 256 \
--output_file /home/ma-user/work/mindformers/wly_all_file/train-10000-fastchat256.mindrecord
```

第四步：模型微调

整体命令如下：

```
# 进入对应目录
cd /home/ma-user/work/mindformers/research/

bash ../scripts/msrun_launcher.sh \
"llama3/run_llama3.py \
--config /home/ma-user/work/mindformers/wly_all_file/run_llama3_8b_8k_800T_A2_64G_lora_dis_256.yaml \
--load_checkpoint /home/ma-user/work/llama3-8B.ckpt \
--auto_trans_ckpt False \
--use_parallel True \
--run_mode finetune \
--train_data /home/ma-user/work/mindformers/wly_all_file/train-10000-fastchat256.mindrecord" 4
```

第五步：模型权重合并

整体命令如下：

```
# 进入对应目录
cd /home/ma-user/work/mindformers/

python mindformers/tools/transform_ckpt.py \
--src_ckpt_strategy /home/ma-user/work/mindformers/research/output/strategy/ \
--src_ckpt_dir /home/ma-user/work/mindformers/research/output/checkpoint/ \
--dst_ckpt_dir /home/ma-user/work/mindformers/research/output/checkpoint/ \
--prefix "new_lora_checkpoint_"
```

第六步：原有能力评估

整体命令如下：
```
# 进入对应目录
cd /home/ma-user/work/mindformers/

python run_mindformer.py \
--config research/llama3/run_llama3_8b_8k_800T_A2_64G_lora_256_base_eval.yaml \
--eval_dataset_dir /home/ma-user/work/mindformers/wly_all_file/squad8192.mindrecord \
--run_mode eval \
--load_checkpoint /home/ma-user/work/mindformers/research/output/checkpoint/rank_0/new_lora_checkpoint_0.ckpt \
--epochs 1 \
--batch_size 1 \
--use_parallel False \
--device_id 0 > result_wly_base_www.txt 2>&1 &

```




