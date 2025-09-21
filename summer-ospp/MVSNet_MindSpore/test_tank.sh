##!/usr/bin/env bash

SAVE_DIR="tank_outputs"

TANK_TESTING='/media/outbreak/68E1-B517/Dataset/TankandTemples/test_offline/'

CKPT_FILE="checkpoints/model_000005.ckpt"


if [ ! -d $SAVE_DIR ]; then
    mkdir -p $SAVE_DIR
fi

CUDA_VISIBLE_DEVICES=0 
python test_tank.py --batch_size=1 --testpath=$TANK_TESTING --numdepth 96 --testlist lists/dtu/test.txt --loadckpt $CKPT_FILE --outdir $SAVE_DIR 



