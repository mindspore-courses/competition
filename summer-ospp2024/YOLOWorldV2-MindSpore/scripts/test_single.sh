anno=$1 # val or minival

# s
## no ft
CUDA_VISIBLE_DEVICES=6 python scripts/test_lvis.py s pretrained_weights/ms-change-yolo_world_v2_s_obj365v1_goldg_pretrain-55b943ea.ckpt 640 --work-dir runs/test/s/640 --anno $anno
