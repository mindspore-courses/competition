anno=$1

# s
## no ft
CUDA_VISIBLE_DEVICES=6 python scripts/test_lvis.py s pretrained_weights/ms-change-yolo_world_v2_s_obj365v1_goldg_pretrain-55b943ea.ckpt 640 --work-dir runs/test/s/640 --anno $anno
# ft
CUDA_VISIBLE_DEVICES=6 python scripts/test_lvis.py s pretrained_weights/ms-change-yolo_world_v2_s_obj365v1_goldg_pretrain_1280ft-fc4ff4f7.ckpt 1280 --work-dir runs/test/s/1280 --anno $anno
# m
## no ft
CUDA_VISIBLE_DEVICES=6 python scripts/test_lvis.py m pretrained_weights/ms-change-yolo_world_v2_m_obj365v1_goldg_pretrain-c6237d5b.ckpt 640 --work-dir runs/test/m/640 --anno $anno
## ft
CUDA_VISIBLE_DEVICES=6 python scripts/test_lvis.py m pretrained_weights/ms-change-yolo_world_v2_m_obj365v1_goldg_pretrain_1280ft-77d0346d.ckpt 1280 --work-dir runs/test/m/1280 --anno $anno
# l
## no ft
CUDA_VISIBLE_DEVICES=6 python scripts/test_lvis.py l pretrained_weights/ms-change-yolo_world_v2_l_obj365v1_goldg_pretrain-a82b1fe3.ckpt 640 --work-dir runs/test/l/640 --anno $anno
## ft
CUDA_VISIBLE_DEVICES=6 python scripts/test_lvis.py l pretrained_weights/ms-change-yolo_world_v2_l_obj365v1_goldg_pretrain_1280ft-9babe3f6.ckpt 1280 --work-dir runs/test/l/1280 --anno $anno
# x
## no ft
CUDA_VISIBLE_DEVICES=6 python scripts/test_lvis.py x pretrained_weights/ms-change-yolo_world_v2_x_obj365v1_goldg_cc3mlite_pretrain-8698fbfa.ckpt  640 --work-dir runs/test/x/640 --anno $anno
## ft
CUDA_VISIBLE_DEVICES=6 python scripts/test_lvis.py x pretrained_weights/ms-change-yolo_world_v2_x_obj365v1_goldg_cc3mlite_pretrain_1280ft-14996a36.ckpt 1280 --work-dir runs/test/x/1280 --anno $anno
# xl
## no ft
CUDA_VISIBLE_DEVICES=6 python scripts/test_lvis.py xl pretrained_weights/ms-change-yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.ckpt 640 --work-dir runs/test/xl/640 --anno $anno