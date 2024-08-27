cd /home/ma-user/work/mindformers/research

bash ../scripts/msrun_launcher.sh \
"llama3/run_llama3.py \
--config /home/ma-user/work/run_llama3_8b_8k_800T_A2_64G_lora_dis_256.yaml \
--load_checkpoint /home/ma-user/work/llama3-8B.ckpt \
--auto_trans_ckpt False \
--use_parallel False \
--run_mode finetune \
--train_data /home/ma-user/work/train-fastchat768.mindrecord" 1