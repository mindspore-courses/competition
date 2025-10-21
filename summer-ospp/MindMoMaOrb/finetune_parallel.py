# ============================================================================
# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Finetuning loop."""

import argparse
import logging
import os
import timeit
from typing import Dict, Optional

import mindspore as ms
from mindspore import nn, ops, context
import mindspore.dataset as ds
from mindspore.communication import init
from mindspore.communication import get_rank, get_group_size

from src import base, pretrained, utils
from src.ase_dataset import AseSqliteDataset, BufferData

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def finetune(
        model: nn.Cell,
        optimizer: nn.Optimizer,
        train_dataloader: ds.GeneratorDataset,
        val_dataloader: ds.GeneratorDataset,
        lr_scheduler: Optional[ms.experimental.optim.lr_scheduler] = None,
        clip_grad: Optional[float] = None,
        log_freq: float = 10,
):
    """Train for a fixed number of steps.

    Args:
        model: The model to optimize.
        optimizer: The optimizer for the model.
        dataloader: A Pytorch Dataloader, which may be infinite if num_steps is passed.
        lr_scheduler: Optional, a Learning rate scheduler for modifying the learning rate.
        clip_grad: Optional, the gradient clipping threshold.
        log_freq: The logging frequency for step metrics.
        device: The device to use for training.
        epoch: The number of epochs the model has been fintuned.

    Returns
        A dictionary of metrics.
    """
    if clip_grad is not None:
        hook_handles = utils.gradient_clipping(model, clip_grad)

    train_metrics = utils.ScalarMetricTracker()
    val_metrics = utils.ScalarMetricTracker()
    val_metrics.update(f"Validation dataset size: {len(val_dataloader)}")

    epoch_metrics = {
        "data_time": 0.0,
        "train_time": 0.0,
    }

    # Get gradient function
    grad_fn = ms.value_and_grad(model.loss, None, optimizer.parameters, has_aux=True)
    grad_reducer = nn.DistributedGradReducer(optimizer.parameters)

    # Define function of one-step training
    def train_step(data, label=None):
        (loss, val_logs), grads = grad_fn(data, label)
        grads = grad_reducer(grads)
        optimizer(grads)
        return loss, val_logs

    step_begin = timeit.default_timer()
    # Get tqdm for the training batches
    # for i, batch in tqdm.tqdm(enumerate(train_dataloader)):
    for i, batch in enumerate(train_dataloader):
        epoch_metrics["data_time"] += timeit.default_timer() - step_begin
        # Reset metrics so that it reports raw values for each step but still do averages on
        # the gradient accumulation.
        if i % log_freq == 0:
            train_metrics.reset()

        # with torch.cuda.amp.autocast(enabled=False):
        model.set_train()
        loss, train_logs = train_step(batch)

        epoch_metrics["train_time"] += timeit.default_timer() - step_begin
        train_metrics.update(epoch_metrics)
        train_metrics.update(train_logs)

        if ops.isnan(loss):
            raise ValueError("nan loss encountered")

        if lr_scheduler is not None:
            lr_scheduler.step()
        step_begin = timeit.default_timer()

    if clip_grad is not None:
        for h in hook_handles:
            h.remove()

    return train_metrics.get_metrics(), val_metrics.get_metrics()


def build_loader(
        dataset_path: str,
        num_workers: int,
        batch_size: int,
        augmentation: Optional[bool] = True,
        target_config: Optional[Dict] = None,
        shuffle: Optional[bool] = True,
        **kwargs,
) -> ds.GeneratorDataset:
    """Builds the train dataloader from a config file.

    Args:
        dataset_path: Dataset path.
        num_workers: The number of workers for each dataset.
        batch_size: The batch_size config for each dataset.
        augmentation: If rotation augmentation is used.
        target_config: The target config.

    Returns:
        The train Dataloader.
    """
    log_train = f"Loading datasets: {dataset_path} with {num_workers} workers. "
    dataset = AseSqliteDataset(
        dataset_path, target_config=target_config, augmentation=augmentation, **kwargs
    )

    log_train += f"Total train dataset size: {len(dataset)} samples"
    logging.info(log_train)

    dataset = BufferData(dataset, shuffle=shuffle)
    rank_id = get_rank()
    rank_size = get_group_size()
    dataloader = [
        [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))] \
            for i in range(0, len(dataset), batch_size)
    ]
    dataloader = [
        base.batch_graphs(
            data[rank_id*len(data)//rank_size : (rank_id+1)*len(data)//rank_size]
        ) for data in dataloader
    ]

    return dataloader


def run(args):
    """Training Loop.

    Args:
        config (DictConfig): Config for training loop.
    """
    utils.seed_everything(args.random_seed)

    # load dataset
    train_loader = build_loader(
        dataset_path=args.train_data_path,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        target_config={"graph": ["energy", "stress"], "node": ["forces"]},
        augmentation=True,
    )
    val_loader = build_loader(
        dataset_path=args.val_data_path,
        num_workers=args.num_workers,
        batch_size=1000, # set a big value s.t. we can infer val_dataloader in one step for to obtain val loss easier
        target_config={"graph": ["energy", "stress"], "node": ["forces"]},
        augmentation=False, # do not apply random augment
        shuffle=False,
    )
    num_steps = len(train_loader)

    # Instantiate model
    pretrained_weights_path = os.path.join(args.checkpoint_path, "orb-mptraj-only-v2.ckpt")
    model = pretrained.orb_mptraj_only_v2(pretrained_weights_path)
    model_params = sum(p.size for p in model.trainable_params() if p.requires_grad)
    logging.info("Model has %d trainable parameters.", model_params)

    total_steps = args.max_epochs * num_steps
    optimizer, lr_scheduler = utils.get_optim(args.lr, total_steps, model)

    start_epoch = 0
    train_time = timeit.default_timer()
    for epoch in range(start_epoch, args.max_epochs):
        train_metrics, _ = finetune(
            model=model,
            optimizer=optimizer,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            lr_scheduler=lr_scheduler,
            clip_grad=args.gradient_clip_val,
        )
        print(f'Epoch: {epoch}/{args.max_epochs}, \n train_metrics: {train_metrics}')

        # Save checkpoint from last epoch
        if epoch == args.max_epochs - 1:
            # create ckpts folder if it does not exist
            if not os.path.exists(args.checkpoint_path):
                os.makedirs(args.checkpoint_path)
            rank_id = get_rank()
            rank_size = get_group_size()
            ms.save_checkpoint(
                model,
                os.path.join(
                    args.checkpoint_path,
                    f"orb-ft-parallel[{rank_id}-{rank_size}]-checkpoint_epoch{epoch}.ckpt"
                ),
            )
            logging.info("Checkpoint saved to %s", args.checkpoint_path)
    logging.info("Training time: %.5f seconds", timeit.default_timer() - train_time)


def main():
    """Main."""
    parser = argparse.ArgumentParser(
        description="Finetune orb model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, default="configs/config_parallel.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="PYNATIVE",
        choices=["GRAPH", "PYNATIVE"],
        help="Context mode, support 'GRAPH', 'PYNATIVE'"
    )
    parser.add_argument(
        "--device_target",
        type=str,
        default="Ascend",
        help="The target device to run, support 'Ascend'"
    )
    args = parser.parse_args()
    ms.set_context(
        mode=context.GRAPH_MODE if args.mode.upper().startswith("GRAPH") else context.PYNATIVE_MODE,
        device_target=args.device_target,
        pynative_synchronize=True,
    )
    # Set parallel context
    ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL, gradients_mean=True)
    init()
    ms.set_seed(1)

    args = utils.load_cfg(args.config)
    run(args)


if __name__ == "__main__":
    main()
