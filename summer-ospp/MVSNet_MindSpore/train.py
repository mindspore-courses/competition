# -*- coding: utf-8 -*-
import argparse
import os
import numpy as np
import time
import datetime
import sys

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import context, Tensor
from mindspore.train.callback import LossMonitor, TimeMonitor
from mindspore.train.serialization import save_checkpoint, load_checkpoint, load_param_into_net
from mindspore.train.model import Model
from mindspore.dataset import GeneratorDataset
from mindspore import dtype as mstype
from mindspore.ops import composite as C
# 导入自定义模块
from datasets.dtu_yao import MVSDataset
from models import MVSNet, mvsnet_loss
from utils import *


# 设置 MindSpore 运行环境
# context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend", device_id=0)

print(mindspore.__version__)
print(mindspore.get_context("device_target"))  # 默认 "GPU"
mindspore.context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

parser = argparse.ArgumentParser(description='A MindSpore Implementation of MVSNet')
parser.add_argument('--mode', default='train', help='train or test', choices=['train', 'test'])
parser.add_argument('--model', default='mvsnet', help='select model')

parser.add_argument('--dataset', default='dtu_yao', help='select dataset')
parser.add_argument('--trainpath', help='train datapath')
parser.add_argument('--testpath', help='test datapath')
parser.add_argument('--trainlist', help='train list')
parser.add_argument('--testlist', help='test list')

parser.add_argument('--epochs', type=int, default=16, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.0003, help='learning rate')
parser.add_argument('--lrepochs', type=str, default="10,12,14:2", help='epoch ids to downscale lr and the downscale rate')
parser.add_argument('--wd', type=float, default=0.0, help='weight decay')

parser.add_argument('--batch_size', type=int, default=12, help='train batch size')
parser.add_argument('--numdepth', type=int, default=192, help='the number of depth values')
parser.add_argument('--interval_scale', type=float, default=1.06, help='the number of depth values')

parser.add_argument('--loadckpt', default=None, help='load a specific checkpoint')
parser.add_argument('--logdir', default='./checkpoints/', help='the directory to save checkpoints/logs')
parser.add_argument('--resume', action='store_true', help='continue to train the model')

parser.add_argument('--summary_freq', type=int, default=20, help='print and summary frequency')
parser.add_argument('--save_freq', type=int, default=1, help='save checkpoint frequency')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')

args = parser.parse_args()
if args.resume:
    assert args.mode == "train"
    assert args.loadckpt is None
if args.testpath is None:
    args.testpath = args.trainpath

# 设置随机种子
np.random.seed(args.seed)
mindspore.set_seed(args.seed)

# log
if args.mode == "train":
    os.makedirs(args.logdir, exist_ok=True)
    print("current time", datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
print("argv:", sys.argv[1:])
print_args(args)

# dataset
train_dataset = MVSDataset(args.trainpath, args.trainlist, mode="train", nviews=3,
                           ndepths=args.numdepth, interval_scale=args.interval_scale)
test_dataset = MVSDataset(args.testpath, args.testlist, mode="test", nviews=5,
                          ndepths=args.numdepth, interval_scale=args.interval_scale)

train_loader = GeneratorDataset(train_dataset,
    column_names=["imgs", "proj_matrices", "depth", "depth_values", "mask", "viewid", "scanid"],
    shuffle=True).batch(batch_size=args.batch_size)

test_loader = GeneratorDataset(test_dataset,
    column_names=["imgs", "proj_matrices", "depth", "depth_values", "mask", "viewid", "scanid"],
    shuffle=False).batch(batch_size=args.batch_size)

# model
model = MVSNet(refine=False)
loss_fn = mvsnet_loss
# optimizer
optimizer = nn.Adam(model.trainable_params(), learning_rate=args.lr, beta1=0.9, beta2=0.999, weight_decay=args.wd)

# loss cell
class WithLossCell(nn.Cell):
    def __init__(self, backbone, loss_fn):
        super(WithLossCell, self).__init__(auto_prefix=False)
        self.backbone = backbone
        self.loss_fn = loss_fn

    def construct(self, imgs, proj_matrices, depth_values, depth_gt, mask):
        outputs = self.backbone(imgs, proj_matrices, depth_values)
        depth_est = outputs["depth"]
        loss = self.loss_fn(depth_est, depth_gt, mask)
        return loss

train_network = WithLossCell(model, loss_fn)
train_model = nn.TrainOneStepCell(train_network, optimizer)
train_model.set_train()

start_epoch = 0
if (args.mode == "train" and args.resume) or (args.mode == "test" and not args.loadckpt):
    ckpts = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
    if ckpts:
        ckpts = sorted(ckpts, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        loadckpt = os.path.join(args.logdir, ckpts[-1])
        print("resuming", loadckpt)
        param_dict = load_checkpoint(loadckpt)
        try:
            load_param_into_net(model, param_dict)
            print("Loaded params into model from", loadckpt)
        except Exception as e:
            print("Warning: load_param_into_net failed for model:", e)
        try:
            load_param_into_net(train_model, param_dict)
            print("Attempted to load params into train_model (may include optimizer state).")
        except Exception:
            pass
        start_epoch = int(ckpts[-1].split('_')[-1].split('.')[0]) + 1
elif args.loadckpt:
    print("loading model {}".format(args.loadckpt))
    param_dict = load_checkpoint(args.loadckpt)
    try:
        load_param_into_net(model, param_dict)
        print("Loaded params into model from", args.loadckpt)
    except Exception as e:
        print("Warning: load_param_into_net failed for model:", e)

print("start at epoch {}".format(start_epoch))

# 学习率调度器
def adjust_learning_rate(epoch):
    parts = args.lrepochs.split(':')
    milestones = [int(x) for x in parts[0].split(',') if x != '']
    decay = float(parts[1])
    n_decays = sum(epoch >= m for m in milestones)
    factor = (1.0 / decay) ** n_decays
    return args.lr * factor

# 在 train() 外部一次性创建 grad operator（节省开销）
_grad_op = C.GradOperation(get_by_list=True, sens_param=False)
_params = model.trainable_params()  # list of Parameter

# 训练
def train():
    global optimizer
    for epoch in range(start_epoch, args.epochs):
        current_lr = adjust_learning_rate(epoch)
        optimizer.learning_rate.set_data(Tensor(current_lr, mstype.float32))
        print(f"\nEpoch {epoch}, lr = {current_lr:.6f}")

        train_losses = []
        iter_idx = 0
        for batch_idx, sample in enumerate(train_loader.create_dict_iterator()):
            iter_idx += 1
            imgs = Tensor(sample['imgs'], mstype.float32)
            proj_matrices = Tensor(sample['proj_matrices'], mstype.float32)
            depth_values = Tensor(sample['depth_values'], mstype.float32)
            depth_gt = Tensor(sample['depth'], mstype.float32)
            mask = Tensor(sample['mask'], mstype.float32)
            loss = train_model(imgs, proj_matrices, depth_values, depth_gt, mask)
            # 注意：loss 是 Tensor 标量
            loss_val = float(loss.asnumpy()) if hasattr(loss, 'asnumpy') else float(loss)
            train_losses.append(loss_val)

            if batch_idx % args.summary_freq == 0:
                print(f"Epoch {epoch}/{args.epochs}, Iter {batch_idx}/{len(train_loader)}, train loss = {loss_val:.4f}")
            if batch_idx % 100 == 0:
                ckpt_path = os.path.join(args.logdir, f"model_{epoch:06d}.ckpt")
                save_checkpoint(model, ckpt_path)
                # ---- 该部分检查是否正常训练，梯度下降 ----
                print("=== Grad summary ===")
                grads = _grad_op(train_network, _params)(imgs, proj_matrices, depth_values, depth_gt, mask)
                show_idx = list(range(3)) + list(range(len(_params) - 2, len(_params)))  # 前3层+后2层
                for idx, (p, g) in enumerate(zip(_params, grads)):
                    if g is None or idx not in show_idx:
                        continue
                    g_np = g.asnumpy()
                    print(f"{idx:03d} | {p.name:45s} | mean={g_np.mean():+.3e}, std={g_np.std():.3e}")
                print("=== End grad summary ===")
        # 每个 epoch 保存一次 checkpoint（避免频繁 IO）
        if (epoch + 1) % args.save_freq == 0:
            ckpt_path = os.path.join(args.logdir, f"model_{epoch:06d}.ckpt")
            save_checkpoint(model, ckpt_path)
            print("Saved checkpoint:", ckpt_path)

# 测试/验证
def eval(epoch=0, train_loss=None):
    avg_test_scalars = DictAverageMeter()
    test_losses = []
    model.set_train(False)

    for batch_idx, sample in enumerate(test_loader.create_dict_iterator()):
        imgs = Tensor(sample['imgs'], mstype.float32)
        proj_matrices = Tensor(sample['proj_matrices'], mstype.float32)
        depth_values = Tensor(sample['depth_values'], mstype.float32)
        depth_gt = Tensor(sample['depth'], mstype.float32)
        mask = Tensor(sample['mask'], mstype.float32)

        outputs = model(imgs, proj_matrices, depth_values)
        depth_est = outputs["depth"]
        loss = loss_fn(depth_est, depth_gt, mask)

        test_losses.append(loss.asnumpy())
        scalar_outputs = {"loss": loss}
        scalar_outputs["abs_depth_error"] = AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5)
        scalar_outputs["thres2mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 2)
        scalar_outputs["thres4mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 4)
        scalar_outputs["thres8mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 8)
        avg_test_scalars.update(scalar_outputs)

    print(f"Epoch {epoch} eval: avg test loss={np.mean(test_losses):.4f}, train loss={train_loss}")
    print("metrics:", avg_test_scalars.mean())
    model.set_train(True)

if __name__ == '__main__':
    if args.mode == "train":
        train()
    else:
        eval()
