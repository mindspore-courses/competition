import argparse
import os
import sys
import time
import gc
import datetime

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.dataset import GeneratorDataset
from mindspore import dtype as mstype
from mindspore import context, Tensor
from datasets.dtu_yao import MVSDataset
from mindspore.train.serialization import save_checkpoint, load_checkpoint
from datasets import find_dataset_def
from utils import *
from models import *
from typing import List

parser = argparse.ArgumentParser(description='A PyTorch Implementation of Cascade Cost Volume MVSNet')
parser.add_argument('--model', default='mvsnet', help='select model')
parser.add_argument('--mode', default='train', choices=['train', 'test'])
parser.add_argument('--device_target', default='GPU', choices=['GPU', 'CPU', 'Ascend'])

parser.add_argument('--dataset', default='dtu_yao', help='select dataset')
parser.add_argument('--trainpath', help='train datapath', required=True)
parser.add_argument('--testpath', help='test datapath', required=False)
parser.add_argument('--trainlist', help='train list', required=True)
parser.add_argument('--testlist', help='test list', required=True)
parser.add_argument('--logdir', default='./checkpoints/debug', help='the directory to save checkpoints/logs')
parser.add_argument('--loadckpt', default=None, help='load a specific checkpoint')
parser.add_argument('--resume', action='store_true', help='continue to train the model')

parser.add_argument('--epochs', type=int, default=16, help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=1, help='train batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--wd', type=float, default=0.0, help='weight decay')


parser.add_argument('--ndepths', type=str, default="48,32,8")
parser.add_argument('--cr_base_chs', type=str, default="8,8,8", help='cost regularization base channels,this is [8]*len(ndepths) in fact')
parser.add_argument('--numdepth', type=int, default=192, help='the number of depth values')
parser.add_argument('--depth_inter_r', type=str, default="4,2,1", help='depth_intervals_ratio')
parser.add_argument('--dlossw', type=str, default="0.5,1.0,2.0", help='depth loss weight for different stage')


parser.add_argument('--lrepochs', type=str, default="10,12,14:2", help='epoch ids to downscale lr and the downscale rate')
parser.add_argument('--warmup_iters', type=int, default=500)
parser.add_argument('--warmup_factor', type=float, default=1.0/3)
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--interval_scale', type=float, default=1.06, help='the number of depth values')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')


parser.add_argument('--summary_freq', type=int, default=50, help='print and summary frequency')
parser.add_argument('--save_freq', type=int, default=1, help='save checkpoint frequency')
parser.add_argument('--eval_freq', type=int, default=1, help='eval freq')
parser.add_argument('--share_cr', action='store_true', help='whether share the cost volume regularization')


def build_warmup_multistep_lr(base_lr: float,
                              steps_per_epoch: int,
                              total_epochs: int,
                              milestones: List[int],
                              gamma: float,
                              warmup_factor: float = 1.0/3,
                              warmup_iters: int = 500):
    total_steps = steps_per_epoch * total_epochs
    lr_each_step = np.zeros((total_steps,), dtype=np.float32)
    for step in range(total_steps):
        # warmup
        if step < warmup_iters:
            alpha = step / float(max(1, warmup_iters))
            lr = base_lr * (warmup_factor * (1 - alpha) + alpha)
        else:
            # epoch index
            epoch_idx = step // steps_per_epoch
            factor = 1.0
            for m in milestones:
                if epoch_idx >= m:
                    factor *= gamma
            lr = base_lr * factor
        lr_each_step[step] = lr
    return ms.Tensor(lr_each_step)


class NetWithLossCell(nn.Cell):
    def __init__(self, backbone: nn.Cell, loss_fn, dlossw_list: List[float]):
        super(NetWithLossCell, self).__init__()
        self.backbone = backbone
        self.loss_fn = loss_fn
        self.dlossw = dlossw_list

    def construct(self,
                  imgs,
                  stage1_proj, stage2_proj, stage3_proj,
                  stage1_depth, stage2_depth, stage3_depth,
                  stage1_mask, stage2_mask, stage3_mask,
                  depth_values):
        depth_gt_ms = {"stage1": stage1_depth, "stage2": stage2_depth, "stage3": stage3_depth}
        mask_ms = {"stage1": stage1_mask, "stage2": stage2_mask, "stage3": stage3_mask}

        outputs = self.backbone(imgs, stage1_proj, stage2_proj, stage3_proj, depth_values)
        out = self.loss_fn(outputs, depth_gt_ms, mask_ms, dlossw=self.dlossw)
        if isinstance(out, tuple) or isinstance(out, list):
            loss = out[0]
        else:
            loss = out
        # ensure scalar Tensor
        return loss

def evaluate(model_eval: nn.Cell, test_loader: GeneratorDataset, args):
    model_eval.set_train(False)
    total_loss = 0.0
    count = 0
    # create iterator
    it = test_loader.create_dict_iterator()
    for batch in it:
        # convert to Tensor (assume batch items are numpy arrays)
        imgs = ms.Tensor(batch["imgs"])
        s1p = ms.Tensor(batch["stage1_proj"])
        s2p = ms.Tensor(batch["stage2_proj"])
        s3p = ms.Tensor(batch["stage3_proj"])
        s1d = ms.Tensor(batch["stage1_depth"])
        s2d = ms.Tensor(batch["stage2_depth"])
        s3d = ms.Tensor(batch["stage3_depth"])
        s1m = ms.Tensor(batch["stage1_mask"])
        s2m = ms.Tensor(batch["stage2_mask"])
        s3m = ms.Tensor(batch["stage3_mask"])
        depth_values = ms.Tensor(batch["depth_values"])

        outputs = model_eval(imgs, s1p, s2p, s3p, depth_values)
        loss_out = cas_mvsnet_loss(outputs, {"stage1": s1d, "stage2": s2d, "stage3": s3d},
                                   {"stage1": s1m, "stage2": s2m, "stage3": s3m}, dlossw=[float(x) for x in args.dlossw.split(",") if x])
        # loss_out can be Tensor or (Tensor, ...)
        if isinstance(loss_out, (tuple, list)):
            l = loss_out[0].asnumpy().item()
        else:
            l = loss_out.asnumpy().item()
        total_loss += l
        count += 1
    avg = total_loss / max(1, count)
    print(f"Validation avg loss: {avg:.6f} on {count} batches")
    return avg

def train_loop(train_net: nn.Cell, backbone: nn.Cell, train_loader: GeneratorDataset, test_loader: GeneratorDataset, optimizer, args, start_epoch=0):
    steps_per_epoch = train_loader.get_dataset_size()
    total_steps = steps_per_epoch * args.epochs
    print("Steps per epoch:", steps_per_epoch, "Total steps:", total_steps)

    global_step = 0
    for epoch in range(start_epoch, args.epochs):
        backbone.set_train()
        t0 = time.time()
        it = train_loader.create_dict_iterator()
        for i, batch in enumerate(it):
            imgs = ms.Tensor(batch["imgs"])
            s1p = ms.Tensor(batch["stage1_proj"])
            s2p = ms.Tensor(batch["stage2_proj"])
            s3p = ms.Tensor(batch["stage3_proj"])
            s1d = ms.Tensor(batch["stage1_depth"])
            s2d = ms.Tensor(batch["stage2_depth"])
            s3d = ms.Tensor(batch["stage3_depth"])
            s1m = ms.Tensor(batch["stage1_mask"])
            s2m = ms.Tensor(batch["stage2_mask"])
            s3m = ms.Tensor(batch["stage3_mask"])
            depth_values = ms.Tensor(batch["depth_values"])

            # TrainOneStepCell returns loss tensor
            loss = train_net(imgs, s1p, s2p, s3p, s1d, s2d, s3d, s1m, s2m, s3m, depth_values)
            loss_val = loss.asnumpy().item()
            if global_step % 50 == 0:
                print(f"[Epoch {epoch}/{args.epochs}] Step {i}/{steps_per_epoch} GlobalStep {global_step} Loss {loss_val:.6f}")
            global_step += 1

        epoch_time = time.time() - t0
        print(f"Epoch {epoch} done in {epoch_time:.2f}s")

        # save checkpoint
        if (epoch + 1) % 1 == 0:
            save_checkpoint(backbone, epoch, args.logdir)
        gc.collect()
        # eval
        if args.testlist:
            evaluate(backbone, test_loader, args)


if __name__ == '__main__':    
    print(mindspore.__version__)
    print(mindspore.get_context("device_target"))  # 默认 "GPU"
    mindspore.context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    mindspore.context.set_context(memory_optimize_level='O1')
    
    # parse arguments and check
    args = parser.parse_args()
    if args.resume:
        assert args.mode == "train"
        assert args.loadckpt is None
    if args.testpath is None:
        args.testpath = args.trainpath
    set_random_seed(args.seed)

    if args.mode == "train":
        if not os.path.isdir(args.logdir):
            os.makedirs(args.logdir)
        current_time_str = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
        print("current time", current_time_str)
    print("argv:", sys.argv[1:])
    print_args(args)

    # model, optimizer
    model = CascadeMVSNet(refine=False, 
                        ndepths=[int(nd) for nd in args.ndepths.split(",") if nd],
                        depth_interals_ratio=[float(d_i) for d_i in args.depth_inter_r.split(",") if d_i],
                        share_cr=args.share_cr,
                        cr_base_chs=[int(ch) for ch in args.cr_base_chs.split(",") if ch],
                        grad_method="detach")
    model_loss = cas_mvsnet_loss
    # load parameters
    start_epoch = 0
    if args.resume:
        saved_models = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
        saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        loadckpt = os.path.join(args.logdir, saved_models[-1])
        print("resuming", loadckpt)
        load_checkpoint(loadckpt, net=model)
        start_epoch = int(saved_models[-1].split('_')[-1].split('.')[0]) + 1
    elif args.loadckpt:
        print("loading model {}".format(args.loadckpt))
        load_checkpoint(args.loadckpt, net=model)
    print("start at epoch {}".format(start_epoch))
    print('Number of model parameters: {}'.format(len(model.trainable_params())))
    train_dataset = MVSDataset(args.trainpath, args.trainlist, mode="train", nviews=3,ndepths=args.numdepth, interval_scale=args.interval_scale)
    test_dataset = MVSDataset(args.trainpath, args.testlist, mode="test", nviews=5, ndepths=args.numdepth, interval_scale=args.interval_scale)

    train_loader = GeneratorDataset(train_dataset,
        column_names=[
            "imgs", "stage1_proj", "stage2_proj", "stage3_proj",
            "stage1_depth", "stage2_depth", "stage3_depth",
            "stage1_mask", "stage2_mask", "stage3_mask",
            "depth_values", "scanid", "viewid"
        ],
        shuffle=True).batch(batch_size=args.batch_size)

    test_loader = GeneratorDataset(test_dataset,
        column_names=[
            "imgs", "stage1_proj", "stage2_proj", "stage3_proj",
            "stage1_depth", "stage2_depth", "stage3_depth",
            "stage1_mask", "stage2_mask", "stage3_mask",
            "depth_values", "scanid", "viewid"
            ],
        shuffle=False).batch(batch_size=args.batch_size)
    steps_per_epoch = train_loader.get_dataset_size()
    parts = args.lrepochs.split(':')
    milestones = [int(x) for x in parts[0].split(',') if x]
    gamma = 1.0 / float(parts[1]) if len(parts) > 1 else 0.1

    lr_tensor = build_warmup_multistep_lr(base_lr=args.lr,
                                         steps_per_epoch=steps_per_epoch,
                                         total_epochs=args.epochs,
                                         milestones=milestones,
                                         gamma=gamma,
                                         warmup_factor=args.warmup_factor,
                                         warmup_iters=args.warmup_iters)

    opt = nn.Adam(params=model.trainable_params(), learning_rate=lr_tensor, beta1=0.9, beta2=0.999, weight_decay=args.wd)
    # NetWithLossCell + TrainOneStepCell
    dlossw = [float(x) for x in args.dlossw.split(",") if x]
    net_with_loss = NetWithLossCell(model, cas_mvsnet_loss, dlossw)
    train_net = nn.TrainOneStepCell(net_with_loss, opt)
    train_net.set_train()
    # run training
    if args.mode == 'train':
        train_loop(train_net, model, train_loader, test_loader, opt, args, start_epoch=start_epoch)
    elif args.mode == 'test':
        if test_loader is None:
            raise RuntimeError("No test loader provided")
        evaluate(model, test_loader, args)
    else:
        raise NotImplementedError