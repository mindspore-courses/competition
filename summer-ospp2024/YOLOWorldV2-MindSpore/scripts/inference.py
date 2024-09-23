# Copyright (c) Tencent. All rights reserved.
import cv2
import supervision as sv
# import torch

from mindspore.dataset.transforms.py_transforms import Compose


from yolow.data.transforms import (YOLOResize, LoadAnnotations, LoadImageFromFile, LoadText,
                                   PackDetInputs, PackDetInputs)

from yolow.model import (build_yolov8_backbone, build_yoloworld_data_preprocessor,
                         build_yoloworld_head, build_yoloworld_neck, build_yoloworld_text,
                         build_yoloworld_backbone, build_yoloworld_detector)


import mindspore as ms
import yaml

bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=1)
mask_annotator = sv.MaskAnnotator()

from mindspore.communication.management import init, get_rank, get_group_size
from mindspore import Profiler

import csv
from collections import defaultdict

from tqdm import tqdm

def init_env(cfg):
    """初始化运行时环境."""
    ms.set_seed(cfg["seed"])
    # 如果device_target设置是None，利用框架自动获取device_target，否则使用设置的。
    if cfg["device_target"] != "None":
        # if cfg["device_target"] not in ["Ascend", "GPU", "CPU"]:
        #     raise ValueError(f"Invalid device_target:{cfg["device_target"]}, "
        #                      f"should be in ['None', 'Ascend', 'GPU', 'CPU']")
        ms.set_context(device_target=cfg["device_target"])

    # 配置运行模式，支持图模式和PYNATIVE模式
    # if cfg["context_mode"] not in ["graph", "pynative"]:
    #     raise ValueError(f"Invalid context_mode: {cfg["context_mode"]}, "
    #                      f"should be in ['graph', 'pynative']")
    context_mode = ms.GRAPH_MODE if cfg["context_mode"] == "graph" else ms.PYNATIVE_MODE
    ms.set_context(mode=context_mode)

    cfg["device_target"] = ms.get_context("device_target")

    # 如果是CPU上运行的话，不配置多卡环境
    if cfg["device_target"] == "CPU":
        cfg["device_id"] = 0
        cfg["device_num"] = 1
        cfg["rank_id"] = 0

    # 设置运行时使用的卡
    if hasattr(cfg, "device_id") and isinstance(cfg["device_id"], int):
        ms.set_context(device_id=cfg["device_id"])

    if cfg["device_num"] > 1:
        # init方法用于多卡的初始化，不区分Ascend和GPU，get_group_size和get_rank方法只能在init后使用
        init()
        print("run distribute!", flush=True)
        group_size = get_group_size()
        # if cfg["device_num"] != group_size:
        #     raise ValueError(f"the setting device_num: {cfg["device_num"]} not equal to the real group_size: {group_size}")
        cfg["rank_id"] = get_rank()
        ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL, gradients_mean=True)
        if hasattr(cfg, "all_reduce_fusion_config"):
            ms.set_auto_parallel_context(all_reduce_fusion_config=cfg.all_reduce_fusion_config)
    else:
        cfg["device_num"] = 1
        cfg["rank_id"] = 0
        print("run standalone!", flush=True)


class LabelAnnotator(sv.LabelAnnotator):

    @staticmethod
    def resolve_text_background_xyxy(
        center_coordinates,
        text_wh,
        position,
    ):
        center_x, center_y = center_coordinates
        text_w, text_h = text_wh
        return center_x, center_y, center_x + text_w, center_y + text_h


label_annotator = LabelAnnotator(text_padding=4, text_scale=0.5, text_thickness=1)

class_names = ("person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, "
               "traffic light, fire hydrant, stop sign, parking meter, bench, bird, "
               "cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, "
               "backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, "
               "sports ball, kite, baseball bat, baseball glove, skateboard, "
               "surfboard, tennis racket, bottle, wine glass, cup, fork, knife, "
               "spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, "
               "hot dog, pizza, donut, cake, chair, couch, potted plant, bed, "
               "dining table, toilet, tv, laptop, mouse, remote, keyboard, "
               "cell phone, microwave, oven, toaster, sink, refrigerator, book, "
               "clock, vase, scissors, teddy bear, hair drier, toothbrush")


def colorstr(*input):
    """
        Helper function for style logging
    """
    *args, string = input if len(input) > 1 else ("bold", input[0])
    colors = {"bold": "\033[1m"}

    return "".join(colors[x] for x in args) + f"{string}"


def create_model(model_size, model_file):
    # We have predefined settings in `model/model_cfgs`,
    # including the default architectures for different
    # sizes of YOLO-World models.
    # You can further specify some items via model_args.
    model_args = dict(
        yoloworld_data_preprocessor=dict(),
        yolov8_backbone=dict(),
        yoloworld_text=dict(),
        yoloworld_backbone=dict(),
        yoloworld_neck=dict(),
        yoloworld_head_module=dict(),
        yoloworld_detector=dict(),
    )
    

    # build model
    data_preprocessor = build_yoloworld_data_preprocessor(model_size, args=model_args)
    ms_backbone = build_yolov8_backbone(model_size, args=model_args)
    ms_text_backbone = build_yoloworld_text(model_size, args=model_args)

    ms_yolow_backbone = build_yoloworld_backbone(model_size, ms_backbone, ms_text_backbone, args=model_args)

    ms_neck = build_yoloworld_neck(model_size, args=model_args)
    ms_head = build_yoloworld_head(model_size, args=model_args)

    ms_yoloworld_model = build_yoloworld_detector(
        model_size, ms_yolow_backbone, ms_neck, ms_head, data_preprocessor, args=model_args)

    ms_ckpt_path = model_file
    
    
    ms.load_checkpoint(ms_ckpt_path, ms_yoloworld_model)

    return ms_yoloworld_model

def dict2csv(dict_lst):
    # Step 1: Aggregate the values for each key
    aggregated_data = defaultdict(list)
    for time_dict in dict_lst:
        for key, value in time_dict.items():
            aggregated_data[key].append(value)

    # Step 2: Compute the average for each key
    average_data = {key: sum(values) / len(values) for key, values in aggregated_data.items()}

    # Step 3: Save the average data to a CSV file
    with open('average_times.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Key', 'Average Value'])
        for key, avg_value in average_data.items():
            writer.writerow([key, avg_value])

    print("Average values saved to average_times.csv")

def run_image(
        ms_model,
        input_image,
        max_num_boxes=100,
        score_thr=0.1,
        nms_thr=0.7,
        img_scale=(1280, 1280),
        output_image="./demo_imgs/output.png",
):
    texts = [[t.strip()] for t in class_names.split(",")] + [[" "]]

    ms_pipeline = Compose([
        LoadImageFromFile(),
        YOLOResize(scale=img_scale),
        LoadAnnotations(with_bbox=True),
        LoadText(),
        PackDetInputs(meta_keys=('img_id', 'img_path', 'ori_shape', \
                            'img_shape', 'scale_factor', 'pad_param', 'texts'))
    ])

    for i in tqdm(range(5000)):

        data_info = ms_pipeline(dict(img_id=0, img_path=input_image, texts=texts))[0]
        # data_info = pipeline(dict(img_id=0, img_path=input_image, texts=texts))
        
        # if isinstance(data_info["inputs"], torch.Tensor):
        #     data_info["inputs"] = ms.Tensor(data_info["inputs"].cpu().numpy())

        data_batch = dict(
            inputs=[data_info["inputs"]],
            data_samples=[data_info["data_samples"]],
        )


        # with torch.no_grad():
        # TODO: stop gradiant in mindspore ?
        time_dict_lst = []
        
        ms_output, time_d = ms_model(data_batch)
        time_dict_lst.append(time_d)
    
    dict2csv(time_dict_lst)
    
        
    ms_output = ms_output[0]
    ms_pred_instances = ms_output['pred_instances']

    # mindspore postprocess    
    box_with_score = ms.ops.cat((ms_pred_instances.bboxes, ms_pred_instances.scores.unsqueeze(-1)), axis=1)
    output_boxes, output_idxs, selected_mask = ms.ops.NMSWithMask(nms_thr)(box_with_score)

    keep_idxs = output_idxs[selected_mask]

    new_ms_pred_instances = {
        'scores': [ms_pred_instances.scores[idx] for idx in keep_idxs if ms_pred_instances.scores[idx].float() > score_thr],
        'labels': [ms_pred_instances.labels[idx] for idx in keep_idxs if ms_pred_instances.scores[idx].float() > score_thr],
        'bboxes': [ms_pred_instances.bboxes[idx] for idx in keep_idxs if ms_pred_instances.scores[idx].float() > score_thr]
    }
    
    new_ms_pred_instances = {key: ms.ops.stack(value).asnumpy() for key, value in new_ms_pred_instances.items()}
    ms_pred_instances = new_ms_pred_instances

    if len(ms_pred_instances['scores']) > max_num_boxes:
        indices = ms_pred_instances['scores'].float().topk(max_num_boxes)[1]
        ms_pred_instances = ms_pred_instances[indices]
    ms_output['pred_instances'] = ms_pred_instances

    if 'masks' in ms_pred_instances:
        masks = ms_pred_instances['masks']
    else:
        masks = None

    detections = sv.Detections(
        xyxy=ms_pred_instances['bboxes'], class_id=ms_pred_instances['labels'], confidence=ms_pred_instances['scores'])

    labels = [
        f"{texts[class_id][0]} {confidence:0.2f}"
        for class_id, confidence in zip(detections.class_id, detections.confidence)
    ]

    # label images
    image = cv2.imread(input_image)
    image = bounding_box_annotator.annotate(image, detections)
    image = label_annotator.annotate(image, detections, labels=labels)
    if masks is not None:
        image = mask_annotator.annotate(image, detections)
    cv2.imwrite(output_image.replace('.png', '_ms_.png'), image)
    print(f"Results saved to {colorstr('bold', output_image.replace('.png', '_ms_.png'))}")


def main():

    with open('./env.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    print(config)
    # 初始化运行时环境
    init_env(config)

    # create model
    ms_model = create_model(
        "s",  # [s/m/l/x/xl]
        "pretrained_weights/ms-change-yolo_world_v2_s_obj365v1_goldg_pretrain-55b943ea.ckpt")

    # start inference
    run_image(ms_model, "./demo_imgs/dog.jpeg")



if __name__ == '__main__':
    main()
