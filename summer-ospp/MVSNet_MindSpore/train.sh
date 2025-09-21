#!/usr/bin/env bash
MVS_TRAINING="/media/outbreak/68E1-B517/Dataset/DTU_ZIP/dtu_training/mvs_training/dtu_training"
MVS_TRAINING="/share/datasets/DTU/mvs_training/dtu/"
# MVS_TRAINING="/home/ma-user/work/dtu_training"
# python train.py --dataset=dtu_yao --batch_size=2 --resume --trainpath=$MVS_TRAINING --trainlist lists/dtu/train.txt --testlist lists/dtu/test.txt --numdepth=192 --logdir ./checkpoints $@
python train.py --dataset=dtu_yao --batch_size=2 --trainpath=$MVS_TRAINING --trainlist lists/dtu/train.txt --testlist lists/dtu/test.txt --numdepth=192 --logdir ./checkpoints $@
