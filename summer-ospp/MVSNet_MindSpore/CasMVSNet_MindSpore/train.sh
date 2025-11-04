#!/usr/bin/env bash
MVS_TRAINING="/media/outbreak/68E1-B517/Dataset/DTU_ZIP/dtu_training/mvs_training/dtu_training"

LOG_DIR="checkpoints"

if [ ! -d $LOG_DIR ]; then
    mkdir -p $LOG_DIR
fi

python train.py --logdir $LOG_DIR --dataset=dtu_yao --resume --batch_size=1 --trainpath=$MVS_TRAINING --trainlist lists/dtu/train.txt --testlist lists/dtu/test.txt --numdepth=64
