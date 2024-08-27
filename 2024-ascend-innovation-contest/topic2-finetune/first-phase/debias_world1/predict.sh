cd /home/ma-user/work/mindformers/research

python llama3/run_llama3_test.py \
--config llama3/run_llama3_8b_8k_800T_A2_64G_lora_256_eval.yaml \
--max_length 768 \
--run_mode predict \
--use_parallel False \
--load_checkpoint /home/ma-user/work/mindformers/research/output/checkpoint/rank_0/new_lora_checkpoint_0.ckpt \
--vocab_file /home/ma-user/work/tokenizer.model \
--auto_trans_ckpt False \
--input_dir "/home/ma-user/work/valid-data-list.json" > valid_data.log 2>&1 &