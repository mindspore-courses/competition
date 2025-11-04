#!/usr/bin/env bash
DTU_TESTING="/data2/local_userdata/outbreak/dtu_test"
CKPT_FILE="/home/outbreak/MVSNet_MindSporeNew/checkpoints/model_000007.ckpt"
OUT_DIR="/data2/local_userdata/outbreak/MVSResultCollection/MVSNetMindsporeTestUseNewHOMOAll"
CUDA_VISIBLE_DEVICES=0
python eval.py --dataset=dtu_yao_eval --batch_size=1 \
    --testpath=$DTU_TESTING --testlist lists/dtu/test.txt \
    --outdir $OUT_DIR \
    --loadckpt $CKPT_FILE $@

