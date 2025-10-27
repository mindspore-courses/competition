import os

from typing import Tuple

import mindspore as ms

from models.dpft import build_dpft

def build(model: str, *args, **kwargs):
    if model == 'dpft':
        return build_dpft(*args, **kwargs)


def load(checkpoint: str, *args, **kwargs) -> Tuple[ms.nn.Cell, int, str]:
    filename = os.path.splitext(os.path.basename(checkpoint))[0]
    timestamp, _, epoch = filename.split('_')
    return ms.load(checkpoint), int(epoch), timestamp