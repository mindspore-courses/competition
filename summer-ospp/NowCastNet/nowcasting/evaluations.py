import mindspore as ms
import mindspore.ops as ops
import mindspore.numpy as mnp
from mindspore import Tensor
from typing import List, Tuple, Dict
import numpy as np

def _to_tensor(x):
    """Convert numpy array or other types to MindSpore tensor"""
    if isinstance(x, np.ndarray):
        return Tensor(x, dtype=ms.float32)
    if isinstance(x, Tensor):
        return x.astype(ms.float32)
    return Tensor(x, dtype=ms.float32)

def _to_numpy(x):
    """Convert MindSpore tensor to numpy array"""
    if isinstance(x, Tensor):
        return x.asnumpy()
    return x

def _ensure_tensor(x):
    """Ensure input is a MindSpore tensor"""
    if isinstance(x, np.ndarray):
        x = Tensor(x, dtype=ms.float32)
    elif not isinstance(x, Tensor):
        x = Tensor(x, dtype=ms.float32)
    return x.astype(ms.float32)

def mae(pred, target, pool_size=None):
    """Mean Absolute Error"""
    pred = _ensure_tensor(pred)
    target = _ensure_tensor(target)
    pred = _maybe_pool(pred, pool_size)
    target = _maybe_pool(target, pool_size)
    
    abs_diff = ops.abs(pred - target)
    return ops.mean(abs_diff)

def mse(pred, target, pool_size=None):
    """Mean Squared Error"""
    pred = _ensure_tensor(pred)
    target = _ensure_tensor(target)
    pred = _maybe_pool(pred, pool_size)
    target = _maybe_pool(target, pool_size)
    
    squared_diff = ops.square(pred - target)
    return ops.mean(squared_diff)

def _maybe_pool(x: Tensor, pool_size: int = None) -> Tensor:
    if pool_size is not None and pool_size > 1:
        pool2d = ops.MaxPool(kernel_size=pool_size, strides=pool_size)
        
        if x.ndim == 3:
            # (H, W, C) -> (1, C, H, W) -> pool -> (C, H, W) -> (H, W, C)
            x = x.transpose((2, 0, 1)).expand_dims(0)
            x = pool2d(x)
            x = x.squeeze(0).transpose((1, 2, 0))
        elif x.ndim == 4:
            # (N, H, W, C) -> (N, C, H, W) -> pool -> (N, C, H, W) -> (N, H, W, C)
            x = x.transpose((0, 3, 1, 2))
            x = pool2d(x)
            x = x.transpose((0, 2, 3, 1))
    return x

def evaluate_all_metrics(pred, target, pool_size=None):
    pred = _ensure_tensor(pred)
    target = _ensure_tensor(target)
    
    metrics = {}
    
    # 基础指标
    metrics['MAE'] = mae(pred, target, pool_size).asnumpy().item()
    metrics['MSE'] = mse(pred, target, pool_size).asnumpy().item()
    metrics['RMSE'] = np.sqrt(metrics['MSE'])
    
    return metrics
