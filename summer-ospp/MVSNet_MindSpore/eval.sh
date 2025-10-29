#!/usr/bin/env bash
DTU_TESTING="/share/datasets/DTU_ZIP/dtu"
CKPT_FILE="checkpoints/model_000001.ckpt"
CUDA_VISIBLE_DEVICES=0
python eval.py --dataset=dtu_yao_eval --batch_size=1 --testpath=$DTU_TESTING --testlist lists/dtu/test.txt --loadckpt $CKPT_FILE $@

