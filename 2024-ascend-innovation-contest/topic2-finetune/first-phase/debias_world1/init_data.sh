cd /home/ma-user/work
wget https://2024-ascend-innovation-contest-mindspore.obs.cn-southwest-2.myhuaweicloud.com/topic2-finetune/train.json
python data_process.py --data_path './train.json' --out_dir './' --train_len 40960  --valid_len 2000
python data_converter.py --data_path './train-data.json' --output_path './train-data-conversation.json'

cd /home/ma-user/work/mindformers
python research/llama3/llama_preprocess.py \
--dataset_type 'qa' \
--input_glob /home/ma-user/work/train-data-conversation.json \
--model_file /home/ma-user/work/tokenizer.model \
--seq_length 768 \
--output_file /home/ma-user/work/train-fastchat768.mindrecord
