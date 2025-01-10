#!/usr/bin/env bash

# init env on cloud (Linux + Ascend), run ~5 min

# uninstall old versions
pip uninstall -y mindformers mindspore-lite

# install mindspore
pip install mindspore==2.3.0RC2
# or try this if above failed
#wget -nc https://2024-ascend-innovation-contest-mindspore.obs.cn-southwest-2.myhuaweicloud.com/topic2-finetune/mindspore-2.3.0rc2-cp39-cp39-linux_aarch64.whl
#pip install mindspore-2.3.0rc2-cp39-cp39-linux_aarch64.whl

# install mindformers
wget -nc https://2024-ascend-innovation-contest-mindspore.obs.cn-southwest-2.myhuaweicloud.com/topic3-infer/mindformers.zip
if [ ! -d "mindformers" ]; then
  unzip mindformers.zip
fi

# install llm-serving
wget -nc https://2024-ascend-innovation-contest-mindspore.obs.cn-southwest-2.myhuaweicloud.com/topic3-infer/llm-serving.zip
if [ ! -d "llm-serving" ]; then
  unzip llm-serving.zip
fi

# install performance_serving
wget -nc https://2024-ascend-innovation-contest-mindspore.obs.cn-southwest-2.myhuaweicloud.com/topic3-infer/performance_serving.zip
if [ ! -d "performance_serving" ]; then
  unzip performance_serving.zip
fi

# precision verification stuff
wget -nc https://2024-ascend-innovation-contest-mindspore.obs.cn-southwest-2.myhuaweicloud.com/topic3-infer/acc_allclose.py
wget -nc https://2024-ascend-innovation-contest-mindspore.obs.cn-southwest-2.myhuaweicloud.com/topic3-infer/file_npy_base.zip
if [ ! -d "file_npy_base" ]; then
  unzip file_npy_base.zip
fi

# setup envvar
export PYTHONPATH=/home/ma-user/work/mindformers:$PYTHONPATH
export PYTHONPATH=/home/ma-user/work/llm-serving:$PYTHONPATH
export PYTHONPATH=/home/ma-user/work/llm-serving/mindspore_serving:$PYTHONPATH
export GRAPH_OP_RUN=1
export MS_ENABLE_INTERNAL_KERNELS=on
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
cd /home/ma-user/work/llm-serving/
pip install -r requirement.txt
pip install tiktoken

# model weights
cd /home/ma-user/work/
mkdir -p checkpoint_download/llama2/
wget -nc https://2024-ascend-innovation-contest-mindspore.obs.cn-southwest-2.myhuaweicloud.com/topic3-infer/llama2_7b.ckpt -P checkpoint_download/llama2/
wget -nc https://2024-ascend-innovation-contest-mindspore.obs.cn-southwest-2.myhuaweicloud.com/topic3-infer/tokenizer.model -P checkpoint_download/llama2/

echo
echo Done!!
