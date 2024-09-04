cd /home/ma-user/work
File_ckpt="./llama3-8B.ckpt"
File_token="./tokenizer.model"
if [ ! -f "$File_ckpt" ]; then
    wget https://2024-ascend-innovation-contest-mindspore.obs.cn-southwest-2.myhuaweicloud.com/topic2-finetune/llama3-8B.ckpt
fi

if [ ! -f "$File_token" ]; then
    wget https://2024-ascend-innovation-contest-mindspore.obs.cn-southwest-2.myhuaweicloud.com/topic2-finetune/tokenizer.model
fi

pip install mindspore==2.3.0RC2
pip install pandas
pip install tiktoken
cd /home/ma-user/work/mindformers
bash build.sh

#cd /home/ma-user/work
# export PYTHONPATH="${PYTHONPATH}:/home/ma-user/work/mindformers/"
# echo $PYTHONPATH

