# Copyright (c) Tencent. All rights reserved.
import argparse
import logging
import os
import os.path as osp
import time

from yolow.data import build_lvis_testloader
from yolow.engine.eval import LVISMetric
from yolow.logger import setup_logger
from yolow.model import (build_yolov8_backbone, build_yoloworld_backbone, build_yoloworld_data_preprocessor,
                         build_yoloworld_detector, build_yoloworld_head, build_yoloworld_neck, build_yoloworld_text)
import mindspore as ms
import yaml



def parse_args():
    parser = argparse.ArgumentParser(description='YOLO-World test (and eval)')
    parser.add_argument('model_size', choices=['n', 's', 'm', 'l', 'x', 'xl'], help='model size')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('img_size', type=int, help='image size')
    parser.add_argument('--work-dir', help='the directory to save the file containing evaluation metrics')
    parser.add_argument('--anno', choices=['minival', 'val'],help='minival or val')
    parser.add_argument(
        '--deterministic', action='store_true', help='Whether to set the deterministic option for CUDNN backend')
    args = parser.parse_args()
    return args


def create_model(args, logger):
    # We have predefined settings in `model/model_cfgs`,
    # including the default architectures for different
    # sizes of YOLO-World models.
    # You can further specify some items via model_args.
    model_args = dict(
        yoloworld_data_preprocessor=dict(),
        # deepen_factor, widen_factor
        yolov8_backbone=dict(),
        yoloworld_text=dict(),
        # with_text_model
        yoloworld_backbone=dict(),
        yoloworld_neck=dict(),
        # use_bn_head
        yoloworld_head_module=dict(),
        # num_train_classes, num_test_classes
        yoloworld_detector=dict(),
    )

    # test build model
    logger.info(f'Building yolo_world_{args.model_size} model')
    data_preprocessor = build_yoloworld_data_preprocessor(args.model_size, args=model_args)
    yolov8_backbone = build_yolov8_backbone(args.model_size, args=model_args)
    text_backbone = build_yoloworld_text(args.model_size, args=model_args)
    yolow_backbone = build_yoloworld_backbone(args.model_size, yolov8_backbone, text_backbone, args=model_args)
    yolow_neck = build_yoloworld_neck(args.model_size, args=model_args)
    yolow_head = build_yoloworld_head(args.model_size, args=model_args)
    yoloworld_model = build_yoloworld_detector(
        args.model_size, yolow_backbone, yolow_neck, yolow_head, data_preprocessor, args=model_args)

    # test load ckpt (mandatory)
    logger.info(f'Loading checkpoint from {osp.abspath(args.checkpoint)}')
    ms.load_checkpoint(args.checkpoint, yoloworld_model)
    return yoloworld_model


def init_env(cfg):
    from mindspore.communication.management import init, get_rank, get_group_size
    """初始化运行时环境."""
    ms.set_seed(cfg["seed"])
    # 如果device_target设置是None，利用框架自动获取device_target，否则使用设置的。
    if cfg["device_target"] != "None":
        ms.set_context(device_target=cfg["device_target"])

    # 配置运行模式，支持图模式和PYNATIVE模式
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

        cfg["rank_id"] = get_rank()
        ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL, gradients_mean=True)
        if hasattr(cfg, "all_reduce_fusion_config"):
            ms.set_auto_parallel_context(all_reduce_fusion_config=cfg.all_reduce_fusion_config)
    else:
        cfg["device_num"] = 1
        cfg["rank_id"] = 0
        print("run stand alone!", flush=True)

def run_test(args, name, log_period=20):
    # launch distributed process

    with open('./env.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    print(config)
    # 初始化运行时环境
    init_env(config)

    # setup logger
    logger = logging.getLogger('yolow')
    setup_logger(output=osp.join(args.work_dir, name, f'{name}.log'))

    # create model
    model = create_model(args, logger)
    model.set_train(False)
    model.set_grad(False)

    # create test dataloader
    # TODO: support more datasets
    logger.info('Loading LVIS dataloader')

    anno_file = 'lvis/lvis_v1_minival_inserted_image_name.json' if args.anno == 'minival' else 'lvis/lvis_v1_val.json'

    dataloader = build_lvis_testloader(
        img_scale=(args.img_size, args.img_size),
        anno_file=anno_file,)

    # create test evaluator

    logger.info(f'Building LVIS evaluator from {osp.abspath(os.path.join("data/coco", anno_file))}')
    evaluator = LVISMetric(
        ann_file=os.path.join("data/coco", anno_file), metric='bbox', format_only=False, outfile_prefix=f'{osp.join(args.work_dir, name)}/results')

    # test loop
    logger.info(f'Start testing (LEN: {len(dataloader)})')
    # model.eval()

    start = time.perf_counter()
    print("test start")

    for idx, data in enumerate(dataloader):
        data_time = time.perf_counter() - start
        outputs, time_dict = model(data)
        evaluator.process(data_samples=outputs, data_batch=data)
        iter_time = time.perf_counter() - start
        if (idx % log_period == 0):
            logger.info(f'TEST [{idx}/{len(dataloader)}]\t'
                        f'iter_time: {iter_time:.4f}\t'
                        f'data_time: {data_time:.4f}\t'
                        )
        start = time.perf_counter()


    # compute metrics
    metrics = evaluator.evaluate(len(dataloader.dataset))
    logger.info(metrics)

    return metrics


def main():
    # parse arguments
    args = parse_args()
    if args.work_dir is None:
        args.work_dir = f'./work_dirs/yolo_world_{args.model_size}_test'

    # create work_dir
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    os.makedirs(osp.join(osp.expanduser(args.work_dir), timestamp), mode=0o777, exist_ok=True)

    # start testing
    run_test(args, timestamp)


if __name__ == '__main__':
    main()
