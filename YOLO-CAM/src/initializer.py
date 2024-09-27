# 参数初始化
import math
import mindspore
from mindspore import nn


def default_recurisive_init(custom_cell):
    """Initialize parameter."""
    for _, cell in custom_cell.cells_and_names():
        if isinstance(cell, (nn.Conv2d, nn.Dense)):
            cell.weight.set_data(mindspore.common.initializer.initializer(mindspore.common.initializer.HeUniform(math.sqrt(5)),
                                                                   cell.weight.shape, cell.weight.dtype))


def load_yolov5_params(args, network):
    """Load yolov5 backbone parameter from checkpoint."""
    if args.resume_yolov5:
        param_dict = mindspore.load_checkpoint(args.resume_yolov5)
        param_dict_new = {}
        for key, values in param_dict.items():
            if key.startswith('moments.'):
                continue
            elif key.startswith('yolo_network.'):
                param_dict_new[key[13:]] = values
                args.logger.info('in resume {}'.format(key))
            else:
                param_dict_new[key] = values
                args.logger.info('in resume {}'.format(key))

        args.logger.info('resume finished')
        mindspore.load_param_into_net(network, param_dict_new)
        args.logger.info('load_model {} success'.format(args.resume_yolov5))

    if args.pretrained_checkpoint:
        param_dict = mindspore.load_checkpoint(args.pretrained_checkpoint)
        param_dict_new = {}
        for key, values in param_dict.items():
            if key.startswith('moments.'):
                continue
            elif key.startswith('yolo_network.') and key[13:] in args.checkpoint_filter_list:
                args.logger.info('remove {}'.format(key))
                continue
            elif key.startswith('yolo_network.'):
                param_dict_new[key[13:]] = values
                args.logger.info('in load {}'.format(key))
            else:
                param_dict_new[key] = values
                args.logger.info('in load {}'.format(key))

        args.logger.info('pretrained finished')
        mindspore.load_param_into_net(network, param_dict_new)
        args.logger.info('load_model {} success'.format(args.pretrained_backbone))

    if args.pretrained_backbone:
        param_dict = mindspore.load_checkpoint(args.pretrained_backbone)
        param_dict_new = {}
        for key, values in param_dict.items():
            if key.startswith('moments.'):
                continue
            elif key.startswith('yolo_network.'):
                param_dict_new[key[13:]] = values
                args.logger.info('in resume {}'.format(key))
            else:
                param_dict_new[key] = values
                args.logger.info('in resume {}'.format(key))

        args.logger.info('pretrained finished')
        mindspore.load_param_into_net(network, param_dict_new)
        args.logger.info('load_model {} success'.format(args.pretrained_backbone))
