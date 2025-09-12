import numpy as np
import mindspore as ms
import mindspore.ops as ops
from mindspore.ops import constexpr
import mindspore
@constexpr
def generate_tensor(t_shape):
    return ms.Tensor(np.ones(t_shape), ms.float32)

def mask_fill(mask, data, num):
    select = ops.Select()
    # replace_tensor = generate_tensor(data.shape)
    # replace_tensor = ms.Tensor(np.ones(data.shape), ms.float32)
    # replace_tensor[:] = num
    replace_tensor = ops.fill(ms.float32, data.shape, num)
    return select(mask, replace_tensor, data.astype(ms.float32))

def adjust_learning_rate(optimizer, parameters, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj=='type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch-1) // 1))}
    elif args.lradj=='type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        optimizer = ms.nn.Adam(parameters, learning_rate=lr)
        print('Updating learning rate to {}'.format(lr))
    return optimizer

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        ms.save_checkpoint(model, path+'/'+'checkpoint.ckpt')
        self.val_loss_min = val_loss

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler:
    def __init__(self):
        self.mean = 0.
        self.std = 1.

    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = ms.Tensor(self.mean, data.dtype) if "mindspore.common.tensor.Tensor" in str(type(data)) else self.mean
        std = ms.Tensor(self.std, data.dtype) if "mindspore.common.tensor.Tensor" in str(type(data)) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = ms.Tensor(self.mean, data.dtype) if "mindspore.common.tensor.Tensor" in str(type(data)) else self.mean
        std = ms.Tensor(self.std, data.dtype) if "mindspore.common.tensor.Tensor" in str(type(data)) else self.std
        if data.shape[-1] != mean.shape[-1]:
            mean = mean[-1:]
            std = std[-1:]
        if not isinstance(self.mean, mindspore.Tensor):
            mean = mindspore.Tensor(mean, dtype=data.dtype)
        if not isinstance(self.std, mindspore.Tensor):
            std = mindspore.Tensor(std, dtype=data.dtype)
        return (data * std) + mean