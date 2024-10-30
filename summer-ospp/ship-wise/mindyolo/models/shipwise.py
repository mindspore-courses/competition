import numpy as np

import mindspore as ms
from mindspore import Tensor, nn

from mindyolo.models.heads.yolov8_head import YOLOv8Head
from mindyolo.models.model_factory import build_model_from_cfg
from mindyolo.models.registry import register_model

__all__ = ["ShipWise", "shipwise"]


def _cfg(url="", **kwargs):
    return {"url": url, **kwargs}


default_cfgs = {"shipwise": _cfg(url="")}


class SEBlock(nn.Cell):
    """Squeeze-and-Excitation Block for channel-wise attention."""

    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2D(1)
        self.fc = nn.SequentialCell(
            nn.Dense(channels, channels // reduction, has_bias=False),
            nn.ReLU(),
            nn.Dense(channels // reduction, channels, has_bias=False),
            nn.Sigmoid()
        )

    def construct(self, x):
        b, c, _, _ = x.shape
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ShipWise(nn.Cell):
    def __init__(self, cfg, in_channels=3, num_classes=None, sync_bn=False):
        super(ShipWise, self).__init__()
        self.cfg = cfg
        self.stride = Tensor(np.array(cfg.stride), ms.int32)
        self.stride_max = int(max(self.cfg.stride))
        ch, nc = in_channels, num_classes

        self.nc = nc  # override yaml value

        # Build the base model
        self.model = build_model_from_cfg(
            model_cfg=cfg, in_channels=ch, num_classes=nc, sync_bn=sync_bn
        )

        # Insert SEBlock into the model without changing input/output interface
        self.insert_se_block()

        self.names = [str(i) for i in range(nc)]  # default names

        self.initialize_weights()

    def construct(self, x):
        return self.model(x)

    def insert_se_block(self):
        """Insert SEBlock into the model's backbone without altering the input/output interface."""
        # Assuming the backbone is a SequentialCell
        backbone = self.model.model[0]
        if isinstance(backbone, nn.SequentialCell):
            # Insert SEBlock after the last layer of the backbone
            layers = list(backbone.cells())
            backbone_out_channels = layers[-1].out_channels
            se_block = SEBlock(channels=backbone_out_channels)

            # Reconstruct the backbone with SEBlock
            new_backbone = nn.SequentialCell(*layers, se_block)
            self.model.model[0] = new_backbone

    def initialize_weights(self):
        # Initialize the weights of SEBlock if present
        backbone = self.model.model[0]
        if isinstance(backbone, nn.SequentialCell):
            for m in backbone.cells():
                if isinstance(m, SEBlock):
                    for layer in m.fc.cells():
                        if isinstance(layer, nn.Dense):
                            ms.common.initializer.initializer(
                                ms.common.initializer.XavierUniform(), layer.weight.shape, layer.weight.dtype
                            )

        # Reset parameters for Detect Head
        m = self.model.model[-1]
        if isinstance(m, YOLOv8Head):
            m.initialize_biases()
            m.dfl.initialize_conv_weight()


@register_model
def shipwise(cfg, in_channels=3, num_classes=None, **kwargs) -> ShipWise:
    """Get ShipWise model."""
    model = ShipWise(cfg=cfg, in_channels=in_channels, num_classes=num_classes, **kwargs)
    return model

# TODO: Preset pre-training model for ShipWise
