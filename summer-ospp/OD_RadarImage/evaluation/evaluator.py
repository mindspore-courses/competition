from __future__ import annotations  # noqa: F407

from typing import Any, Callable, Dict, Iterable, List

import os.path as osp

import mindspore as ms

from tqdm import tqdm
from mindspore import SummaryCollector

from models import load as load_model
from evaluation.exporters import build as build_exporter
from evaluation.metric import build_metric

class CentralizedEvaluator():
    def __init__(self,
                 metric: ms.nn = None,
                 exporter: Callable = None,
                 device: str= None,
                 logging: str= None,
                 ):
        """
        Arguments:
            logging: Logging frenquency. One of either None, step or epoch.
        """
        self.eval_fn = metric
        self.export_fn = exporter
        self.device = device
        self.logging = logging

    @classmethod
    def from_config(cls,
                    config: Dict[str, Any],
                    *args,
                    **kwargs) -> CentralizedEvaluator:  # noqa: F821
        metric = build_metric(
            config['evaluate']
        )
        exporter = build_exporter(config['evaluate']['exporter']['name'], config)
        device = ms.device(config['computing']['device'])
        logging = config['train'].get('logging')

        return cls(
            metric=metric,
            exporter=exporter,
            device=device,
            logging=logging
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.evaluate(*args, **kwargs)

    @staticmethod
    def _dict_to(data: Dict[str, ms.Tensor], device) -> Dict[str, ms.Tensor]:
        return {k: v.to(device) for k, v in data.items()}

    @staticmethod
    def log_scalars(writer, scalars: Dict[str, Any], epoch: int, prefix: str = None) -> None:
        # Get prefix
        prefix = f"{prefix}/" if prefix is not None else ""

        # Add scalar values
        for name, scalar in scalars.items():
            writer.add_scalar(prefix + name, scalar, epoch)

    @ms.no_grad()
    def evalute_complexity(self, epoch:int, model:ms.nn.Cell,
                           data_loader: Iterable, writer=None):
        # Set model to evaluation mode
        model.eval()

        # Get inference test input
        data,_ = next(iter(data_loader))

        # Load test data (to devcie)
        data: Dict[str,ms.tensor] = self._dict_to(data,self.device)

        # Determine model complexity
        with get_accelerator().device(self.device):
            flops, macs, params = get_model_profile(
                model=model, args=(data,),
                print_profile=False, warm_up=10, as_string=False
            )

        # Log model complexity
        self.log_scalars(
            writer, {'FLOPS': flops, 'MACS': macs, 'Parameters': params},
            epoch, 'test'
        )

    @ms.no_grad()
    def evaluate_one_epoch(self, epoch: int, model: ms.nn.Cell,
                           data_loader: Iterable, writer=None, dst: str = None):
        # Set model to evaluation mode
        model.eval()

        # Initialize epoch logs
        scalars = {}

        with tqdm(total=len(data_loader)) as pbar:
            for i, (data, labels) in enumerate(data_loader):
                # Load data and labels (to device)
                labels: List[Dict[str, ms.Tensor]] = \
                    [self._dict_to(label, self.device) for label in labels]
                data: Dict[str, ms.Tensor] = \
                    self._dict_to(data, self.device)

                # Make prediction
                output = model(data)

                # Evaluate model output
                metrics = self.eval_fn(output, labels)

                # Log evaluation step
                if self.logging == 'step':
                    self.log_scalars(writer, metrics, i + epoch * len(data_loader), 'test')

                # Add values to epoch log
                if self.logging == 'epoch':
                    for k, v in metrics.items():
                        scalars[k] = scalars.get(k, 0) + v

                # Export predictions
                if self.export_fn is not None:
                    self.export_fn(output, labels, i * len(labels), dst)

                # Report training progress
                pbar.update()

        if self.logging == 'epoch':
            # Average epoch logs
            scalars = {k: v / (i + 1) for k, v in scalars.items()}

            # Write epoch logs
            self.log_scalars(writer, scalars, epoch, 'test')

    def evaluate(self, checkpoint: str, data_loader: Iterable, dst: str = None):
        # Load model from checkpoint
        model, epoch, timestamp = load_model(checkpoint)

        # Load model (to device)
        model.to(self.device)

        # Check if destination is provided
        if self.logging is not None:
            dst = osp.join(dst, timestamp)

        # Initialize tensorboard writer (logging)
        if self.logging is not None:
            writer = SummaryWriter(log_dir=dst)

        # Evaluate model performance
        self.evaluate_one_epoch(epoch, model, data_loader, writer, dst)

        # Evaluate model inference time
        self.evaluate_inference_time(epoch, model, data_loader, writer)

        # Evaluate model complexity
        self.evaluate_complexity(epoch, model, data_loader, writer)

        # Flush and close writer
        if self.logging is not None:
            writer.flush()
            writer.close()


def build_evaluator(*args, **kwargs):
    return CentralizedEvaluator.from_config(*args, **kwargs)