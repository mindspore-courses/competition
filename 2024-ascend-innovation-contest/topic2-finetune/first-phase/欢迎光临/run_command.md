#### URL链接

wget https://2024-ascend-innovation-contest-mindspore.obs.cn-southwest-2.myhuaweicloud.com/topic2-finetune/mindformers.zip

```
export PYTHONPATH="${PYTHONPATH}:/home/ma-user/work/mindformers/"
```

wget https://2024-ascend-innovation-contest-mindspore.obs.cn-southwest-2.myhuaweicloud.com/topic2-finetune/llama3-8B.ckpt

wget https://2024-ascend-innovation-contest-mindspore.obs.cn-southwest-2.myhuaweicloud.com/topic2-finetune/tokenizer.model

wget https://2024-ascend-innovation-contest-mindspore.obs.cn-southwest-2.myhuaweicloud.com/topic2-finetune/train.json

wget https://2024-ascend-innovation-contest-mindspore.obs.cn-southwest-2.myhuaweicloud.com/topic2-finetune/train-data-conversation.json

wget https://2024-ascend-innovation-contest-mindspore.obs.cn-southwest-2.myhuaweicloud.com/topic2-finetune/train-fastchat256-mindrecore.zip

wget https://2024-ascend-innovation-contest-mindspore.obs.cn-southwest-2.myhuaweicloud.com/topic2-finetune/run_llama3_8b_8k_800T_A2_64G_lora_dis_256.yaml

/home/ma-user/work/mindformers/research/output/msrun_log/



wget https://2024-ascend-innovation-contest-mindspore.obs.cn-southwest-2.myhuaweicloud.com/topic2-finetune/train.json

#### 微调数据

```
wget https://2024-ascend-innovation-contest-mindspore.obs.cn-southwest-2.myhuaweicloud.com/topic2-finetune/train-fastchat256-mindrecore.zip
unzip train-fastchat256-mindrecore.zip

wget https://2024-ascend-innovation-contest-mindspore.obs.cn-southwest-2.myhuaweicloud.com/topic2-finetune/train-data-conversation.json

cd /home/ma-user/work/mindformers/research/llama3
python llama_preprocess.py \
--dataset_type qa \
--input_glob /home/ma-user/work/tmp_data/sampled_data.json \
--model_file /home/ma-user/work/tokenizer.model \
--seq_length 256 \
--output_file /home/ma-user/work/train-fastchat256.mindrecord
```



#### 微调命令

```
cd /home/ma-user/work/mindformers/research/
bash ../scripts/msrun_launcher.sh \
"llama3/run_llama3.py \
--config /home/ma-user/work/run_llama3_8b_8k_800T_A2_64G_lora_dis_256.yaml \
--load_checkpoint /home/ma-user/work/llama3-8B.ckpt \
--auto_trans_ckpt False \
--use_parallel True \
--run_mode finetune \
--train_data /home/ma-user/work/train-fastchat256.mindrecord" 4
```

```
npu-smi info
kill -9 进程号
new_lora_checkpoint_0.ckpt
```

#### 权重合并

```
cd /home/ma-user/work/mindformers/
python mindformers/tools/transform_ckpt.py \
--src_ckpt_strategy /home/ma-user/work/mindformers/research/output/strategy/ \
--src_ckpt_dir /home/ma-user/work/mindformers/research/output/checkpoint/ \
--dst_ckpt_dir /home/ma-user/work/mindformers/research/output/checkpoint/ \
--prefix "new_lora_checkpoint_"
```



#### 启动配置环境

```
pip install mindspore==2.3.0RC2

cd mindformers/
bash build.sh

export PYTHONPATH="${PYTHONPATH}:/home/ma-user/work/mindformers/"

pip install tiktoken
```

#### 测试数据

```
wget https://2024-ascend-innovation-contest-mindspore.obs.cn-southwest-2.myhuaweicloud.com/topic2-finetune/squad1.1.zip


cd /home/ma-user/work/mindformers/mindformers/tools/dataset_preprocess/llama/
python squad_data_process.py \
--input_file  /home/ma-user/work/dev-v1.1.json \
--output_file  /home/ma-user/work/squad8192.mindrecord \
--mode eval \
--max_length 8192 \
--tokenizer_type "llama3-8B" > test_eval_base.log 2>&1 &
```



#### 测试代码

```
python run_mindformer.py \
--config research/llama3/run_llama3_8b_8k_800T_A2_64G_lora_256_base_eval.yaml \
--eval_dataset_dir /home/ma-user/work/squad8192.mindrecord \
--run_mode eval \
--load_checkpoint /home/ma-user/work/checkpoint_0.ckpt \
--epochs 1 \
--batch_size 1 \
--use_parallel False \
--device_id 0
```

#### 运行推理

```
wget https://2024-ascend-innovation-contest-mindspore.obs.cn-southwest-2.myhuaweicloud.com/topic2-finetune/run_llama3_test.py -P /home/ma-user/work/mindformers/research/llama3/

cd /home/ma-user/work/mindformers/research
python llama3/run_llama3_test.py \
--config  llama3/run_llama3_8b_8k_800T_A2_64G_lora_256_eval.yaml \
--run_mode predict \
--use_parallel False \
--load_checkpoint /home/ma-user/work/checkpoint_0.ckpt \
--vocab_file /home/ma-user/work/tokenizer.model \
--auto_trans_ckpt False \
--input_dir "/home/ma-user/work/train.json" > data_test_2000_1.log 2>&1 &
```

```
wget https://2024-ascend-innovation-contest-mindspore.obs.cn-southwest-2.myhuaweicloud.com/topic2-finetune/run_llama3_8b_8k_800T_A2_64G_lora_256_base_eval.yaml -P /home/ma-user/work/mindformers/research/llama3/
```

```
wget https://2024-ascend-innovation-contest-mindspore.obs.cn-southwest-2.myhuaweicloud.com/topic2-finetune/run_llama3_8b_8k_800T_A2_64G_lora_256_eval.yaml -P /home/ma-user/work/mindformers/research/llama3/
```

