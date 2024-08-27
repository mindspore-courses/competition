#合并权重
cd /home/ma-user/work/mindformers

python mindformers/tools/transform_ckpt.py \
--src_ckpt_strategy /home/ma-user/work/mindformers/research/output/strategy/ \
--src_ckpt_dir /home/ma-user/work/mindformers/research/output/checkpoint/ \
--dst_ckpt_dir /home/ma-user/work/mindformers/research/output/checkpoint/ \
--prefix "new_lora_checkpoint_"