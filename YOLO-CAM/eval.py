# YoloV5的评估脚本
import os
import time
import shutil

import mindspore
from mindspore import ParallelMode
from mindspore.communication.management import init, get_group_size, get_rank

from src.yolo import YOLOV5
from src.logger import get_logger
from src.util import DetectionEngine, EvalWrapper
from src.yolo_dataset import create_yolo_dataset

from model_utils.config import config
from model_utils.moxing_adapter import moxing_wrapper, modelarts_pre_process


def eval_preprocess():
    config.val_img_dir = os.path.join(config.data_dir, config.val_img_dir)
    config.val_ann_file = os.path.join(config.data_dir, config.val_ann_file)
    device_id = int(os.getenv('DEVICE_ID', '0'))
    mindspore.set_context(mode=0, device_target=config.device_target, device_id=device_id)
    parallel_mode = ParallelMode.STAND_ALONE
    config.eval_parallel = config.is_distributed and config.eval_parallel
    device_num = 1
    if config.eval_parallel:
        init()
        config.rank = get_rank()
        config.group_size = get_group_size()
        device_num = get_group_size()
        parallel_mode = ParallelMode.DATA_PARALLEL
    mindspore.reset_auto_parallel_context()
    mindspore.set_auto_parallel_context(parallel_mode=parallel_mode, gradients_mean=True, device_num=device_num)

    config.logger = get_logger(config.output_dir, device_id)


def load_parameters(network, filename):
    config.logger.info("yolov5 pretrained network model: %s", filename)
    param_dict = mindspore.load_checkpoint(filename)
    param_dict_new = {}
    for key, values in param_dict.items():
        if key.startswith('moments.'):
            continue
        elif key.startswith('yolo_network.'):
            param_dict_new[key[13:]] = values
        else:
            param_dict_new[key] = values
    mindspore.load_param_into_net(network, param_dict_new)
    config.logger.info('load_model %s success', filename)


@moxing_wrapper(pre_process=modelarts_pre_process, pre_args=[config])
def run_eval():
    eval_preprocess()
    start_time = time.time()
    config.logger.info('Creating Network....')
    dict_version = {'yolov5s': 0, 'yolov5m': 1, 'yolov5l': 2, 'yolov5x': 3}
    network = YOLOV5(is_training=False, version=dict_version[config.yolov5_version])

    if os.path.isfile(config.pretrained):
        load_parameters(network, config.pretrained)
    else:
        raise FileNotFoundError(f"{config.pretrained} is not a filename.")
    rank_id = int(os.getenv('RANK_ID', '0'))
    if config.eval_parallel:
        rank_id = get_rank()
    ds = create_yolo_dataset(config.val_img_dir, config.val_ann_file, is_training=False,
                             batch_size=config.per_batch_size, device_num=config.group_size,
                             rank=rank_id, shuffle=False, config=config)

    config.logger.info('testing shape : %s', config.test_img_shape)
    config.logger.info('total %d images to eval', ds.get_dataset_size() * config.per_batch_size)

    network.set_train(False)

    detection = DetectionEngine(config, config.test_ignore_threshold)
    if config.eval_parallel:
        if os.path.exists(config.save_prefix):
            shutil.rmtree(config.save_prefix, ignore_errors=True)

    config.logger.info('Start inference....')
    eval_wrapper = EvalWrapper(config, network, ds, detection)
    eval_wrapper.inference()
    eval_result, _ = eval_wrapper.get_results()

    cost_time = time.time() - start_time
    eval_log_string = '\n=============coco eval result=========\n' + eval_result
    config.logger.info(eval_log_string)
    config.logger.info('testing cost time %.2f h', cost_time / 3600.)


if __name__ == "__main__":
    run_eval()
