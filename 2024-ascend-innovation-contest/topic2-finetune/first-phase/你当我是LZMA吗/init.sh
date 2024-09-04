#!/usr/bin/env bash

# init env on cloud (Linux + Ascend), run ~5 min

# install mindspore
pip install mindspore==2.3.0RC2
# or try this if above failed
#wget -nc https://2024-ascend-innovation-contest-mindspore.obs.cn-southwest-2.myhuaweicloud.com/topic2-finetune/mindspore-2.3.0rc2-cp39-cp39-linux_aarch64.whl
#pip install mindspore-2.3.0rc2-cp39-cp39-linux_aarch64.whl

# install mindformers
wget -nc https://2024-ascend-innovation-contest-mindspore.obs.cn-southwest-2.myhuaweicloud.com/topic2-finetune/mindformers.zip
if [ ! -d "mindformers" ]; then
  unzip mindformers.zip
  cd mindformers/
  bash build.sh
fi

# setup envvar
export PYTHONPATH="${PYTHONPATH}:/home/ma-user/work/mindformers/"
echo $PYTHONPATH

# make aliases
alias cls=clear
alias copy=cp
alias move=mv
alias ren=mv
alias cd=pushd
alias po=popd
alias py=python
alias k9='kill -9'

# install dependencies
pip install tiktoken

# model weights
cd /home/ma-user/work/
wget -nc https://2024-ascend-innovation-contest-mindspore.obs.cn-southwest-2.myhuaweicloud.com/topic2-finetune/llama3-8B.ckpt
wget -nc https://2024-ascend-innovation-contest-mindspore.obs.cn-southwest-2.myhuaweicloud.com/topic2-finetune/tokenizer.model

# datasets
cd /home/ma-user/work/
# (pretrain)
wget -nc https://2024-ascend-innovation-contest-mindspore.obs.cn-southwest-2.myhuaweicloud.com/topic2-finetune/squad1.1.zip
if [ ! -f "dev-v1.1.json" ]; then
  unzip squad1.1.zip
  rm train-v1.1.json
fi
# (finetune-original)
wget -nc https://2024-ascend-innovation-contest-mindspore.obs.cn-southwest-2.myhuaweicloud.com/topic2-finetune/train.json
# (finetune-reprocessed)
#wget -nc https://2024-ascend-innovation-contest-mindspore.obs.cn-southwest-2.myhuaweicloud.com/topic2-finetune/data_converter.py
#wget -nc https://2024-ascend-innovation-contest-mindspore.obs.cn-southwest-2.myhuaweicloud.com/topic2-finetune/train-data-conversation.json
# (finetune-mindrecord)
#wget -nc https://2024-ascend-innovation-contest-mindspore.obs.cn-southwest-2.myhuaweicloud.com/topic2-finetune/train-fastchat256-mindrecore.zip
#if [ ! -f "train-fastchat256.mindrecord" ]; then
#  unzip train-fastchat256-mindrecore.zip
#fi

# finetune config
#wget -nc https://2024-ascend-innovation-contest-mindspore.obs.cn-southwest-2.myhuaweicloud.com/topic2-finetune/run_llama3_8b_8k_800T_A2_64G_lora_dis_256.yaml
#wget -nc https://2024-ascend-innovation-contest-mindspore.obs.cn-southwest-2.myhuaweicloud.com/topic2-finetune/run_llama3_8b_8k_800T_A2_64G_lora_dis8_256.yaml
# eval config
pushd /home/ma-user/work/mindformers/research/llama3/
wget -nc https://2024-ascend-innovation-contest-mindspore.obs.cn-southwest-2.myhuaweicloud.com/topic2-finetune/run_llama3_test.py
wget -nc https://2024-ascend-innovation-contest-mindspore.obs.cn-southwest-2.myhuaweicloud.com/topic2-finetune/run_llama3_8b_8k_800T_A2_64G_lora_256_base_eval.yaml
wget -nc https://2024-ascend-innovation-contest-mindspore.obs.cn-southwest-2.myhuaweicloud.com/topic2-finetune/run_llama3_8b_8k_800T_A2_64G_lora_256_eval.yaml
popd

echo
echo Done!!
