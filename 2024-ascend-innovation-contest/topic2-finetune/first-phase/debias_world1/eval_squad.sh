cd /home/ma-user/work
File="./squad1.1.zip"
if [ ! -f "$File" ]; then
    wget https://2024-ascend-innovation-contest-mindspore.obs.cn-southwest-2.myhuaweicloud.com/topic2-finetune/squad1.1.zip
    unzip squad1.1.zip
    
    cd /home/ma-user/work/mindformers/mindformers/tools/dataset_preprocess/llama/
    python squad_data_process.py \
    --input_file /home/ma-user/work/dev-v1.1.json \
    --output_file /home/ma-user/work/squad8192.mindrecord \
    --mode eval \
    --max_length 8192 \
    --tokenizer_type "llama3-8B"
fi

cd /home/ma-user/work/mindformers/
python run_mindformer.py \
--config research/llama3/run_llama3_8b_8k_800T_A2_64G_lora_256_base_eval.yaml \
--eval_dataset_dir /home/ma-user/work/squad8192.mindrecord \
--run_mode eval \
--load_checkpoint /home/ma-user/work/mindformers/research/output/checkpoint/rank_0/new_lora_checkpoint_0.ckpt \
--epochs 1 \
--batch_size 1 \
--use_parallel False \
--device_id 0 > eval_squad.log 2>&1 &