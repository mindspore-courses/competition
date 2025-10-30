#!/usr/bin/env bash
MVS_TRAINING="/share/datasets/DTU/mvs_training/dtu/"
CUDA_VISIBLE_DEVICES=0
python train.py --dataset=dtu_yao --batch_size=2 --trainpath=$MVS_TRAINING --trainlist lists/dtu/train.txt --testlist lists/dtu/test.txt --numdepth=192 --logdir ./checkpoints1028 $@
