import numpy as np
import mindspore 
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore.train.summary import SummaryRecord
import mindspore.numpy as mnp

def print_args(args):
    print("################################  args  ################################")
    for k, v in args.__dict__.items():
        print("{0: <10}\t{1: <30}\t{2: <20}".format(k, str(v), str(type(v))))
    print("########################################################################")


def make_nograd_func(func):
    def wrapper(*f_args, **f_kwargs):
        ret = func(*f_args, **f_kwargs)
        return ret
    return wrapper


def make_recursive_func(func):
    def wrapper(vars):
        if isinstance(vars, list):
            return [wrapper(x) for x in vars]
        elif isinstance(vars, tuple):
            return tuple([wrapper(x) for x in vars])
        elif isinstance(vars, dict):
            return {k: wrapper(v) for k, v in vars.items()}
        else:
            return func(vars)
    return wrapper


@make_recursive_func
def tensor2float(vars):
    if isinstance(vars, float):
        return vars
    elif isinstance(vars, mindspore.Tensor):
        return float(vars.asnumpy().item())
    else:
        raise NotImplementedError("invalid input type {} for tensor2float".format(type(vars)))


@make_recursive_func
def tensor2numpy(vars):
    if isinstance(vars, np.ndarray):
        return vars
    elif isinstance(vars, mindspore.Tensor):
        return vars.asnumpy().copy()
    else:
        raise NotImplementedError("invalid input type {} for tensor2numpy".format(type(vars)))


@make_recursive_func
def tocuda(vars):
    return vars

def save_scalars(logger, mode, scalar_dict, global_step):
    scalar_dict = tensor2float(scalar_dict)
    for key, value in scalar_dict.items():
        if not isinstance(value, (list, tuple)):
            name = '{}/{}'.format(mode, key)
            logger.add_scalar(name, value, global_step)
        else:
            for idx in range(len(value)):
                name = '{}/{}_{}'.format(mode, key, idx)
                logger.add_scalar(name, value[idx], global_step)


def make_grid_ms(img_tensor, nrow=1, padding=0, normalize=True):
    """
    img_tensor: shape [N, C, H, W]
    """
    if normalize:
        min_v = mnp.min(img_tensor)
        max_v = mnp.max(img_tensor)
        img_tensor = (img_tensor - min_v) / (max_v - min_v + 1e-5)
    N, C, H, W = img_tensor.shape
    ncol = (N + nrow - 1) // nrow
    grid = mnp.zeros((C, ncol * H, nrow * W))
    for idx in range(N):
        row = idx // nrow
        col = idx % nrow
        grid[:, row * H:(row + 1) * H, col * W:(col + 1) * W] = img_tensor[idx]
    return grid

def save_images(logger: SummaryRecord, mode, images_dict, global_step):
    def preprocess(name, img):
        if not (len(img.shape) == 3 or len(img.shape) == 4):
            raise NotImplementedError(f"invalid img shape {name}:{img.shape} in save_images")

        if len(img.shape) == 3:  # [C,H,W]
            img = img[np.newaxis, ...]   # [1,C,H,W]

        img = mindspore.Tensor(img[:1])
        grid = make_grid_ms(img, nrow=1, padding=0, normalize=True)
        return grid.asnumpy()
    for key, value in images_dict.items():
        if not isinstance(value, (list, tuple)):
            name = f'{mode}/{key}'
            logger.add_value('image', name, value)
            logger.add_image(name, preprocess(name, value), global_step)
        else:
            for idx in range(len(value)):
                name = f'{mode}/{key}_{idx}'
                logger.add_value('image', name, value)
                logger.add_image(name, preprocess(name, value[idx]), global_step)


class DictAverageMeter(object):
    def __init__(self):
        self.data = {}
        self.count = 0

    def update(self, new_input):
        self.count += 1
        if len(self.data) == 0:
            for k, v in new_input.items():
                if not isinstance(v, float):
                    raise NotImplementedError("invalid data {}: {}".format(k, type(v)))
                self.data[k] = v
        else:
            for k, v in new_input.items():
                if not isinstance(v, float):
                    raise NotImplementedError("invalid data {}: {}".format(k, type(v)))
                self.data[k] += v

    def mean(self):
        return {k: v / self.count for k, v in self.data.items()}


# a wrapper to compute metrics for each image individually
def compute_metrics_for_each_image(metric_func):
    def wrapper(depth_est, depth_gt, mask, *args):
        batch_size = depth_gt.shape[0]
        results = []
        # compute result one by one
        for idx in range(batch_size):
            ret = metric_func(depth_est[idx], depth_gt[idx], mask[idx], *args)
            results.append(ret)
        return ops.stack(results).mean()
    return wrapper


@make_nograd_func
@compute_metrics_for_each_image
def Thres_metrics(depth_est, depth_gt, mask, thres):
    assert isinstance(thres, (int, float))
    depth_est_mask, depth_gt_mask = depth_est[mask], depth_gt[mask]
    errors = ops.abs(depth_est_mask - depth_gt_mask)
    err_mask = errors > thres
    return ops.mean(err_mask.astype(mindspore.float32))


# NOTE: please do not use this to build up training loss
@make_nograd_func
@compute_metrics_for_each_image
def AbsDepthError_metrics(depth_est, depth_gt, mask, thres=None):
    depth_est, depth_gt = depth_est[mask], depth_gt[mask]
    error = ops.abs(depth_est - depth_gt)
    if thres is not None:
        error = error[(error >= float(thres[0])) & (error <= float(thres[1]))]
        if error.shape[0] == 0:
            return mindspore.Tensor(0, mindspore.float32)
    return ops.mean(error)
