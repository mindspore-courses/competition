from typing import Dict,Any,OrderedDict
import mindspore as ms
from mindspore import nn,ops
from mindcv.models import create_model
from mindspore import load_checkpoint

class BackboneBase(nn.Cell):
    def __init__(self,
                 backbone:nn.Cell,
                 in_channels:int=3,
                 multi_scale:int=1,
                 channel_last:bool=True,
                 weights:OrderedDict[str,Any]=None,
                 **kwargs):

        """Base class for ResNet backbones with intermediate returns.

        Arguments:
            backbone: Backbone model (e.g., ResNet50 from mindcv).
            in_channels: Input channels (default: 3).
            multi_scale: Number of multi-scale feature maps to return.
            channel_last: If True, input format is (B, H, W, C); else (B, C, H, W).
            weights: Pretrained weights (OrderedDict or checkpoint path).
        """
        super().__init__()
        self.in_channels = in_channels
        self.multi_scale = multi_scale
        self.channel_last = channel_last

        # Channel adjustment layer
        if in_channels == 3:
            self.adjustment_layer = nn.Identity()
        else:
            self.adjustment_layer = nn.Conv2d(in_channels, 3, kernel_size=1, stride=1, 
                                             pad_mode='pad', padding=0, has_bias=False)
        
        # Configure return layers (e.g., {'layer1': '1', 'layer2': '2'})
        self.return_layers = {f'layer{i+1}':str(i+1) for i in range(multi_scale)}

        self.body = nn.CellList()

        conv1=nn.Conv2d(in_channels=3,out_channels=64,kernel_size=(7,7),stride=(2,2),pad_mode='pad',padding=3)
        bn1=nn.BatchNorm2d(num_features=64,eps=1e-05,momentum=0.1)
        relu=nn.ReLU()
        maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1,dilation=1,ceil_mode=False,pad_mode='pad')

        seq=nn.SequentialCell([conv1,bn1,relu,maxpool])

        self.body.append(seq)

        for name,module in backbone.cells_and_names():
            for layer in self.return_layers:
                if name == layer:
                    self.body.append(module)


        # Load weights if provided
        if weights:
            if isinstance(weights,str):
                load_checkpoint(weights,self)
            else:
                ms.load_param_into_net(self,weights)


    @property
    def multi_scale(self):
        return self._multi_scale

    @multi_scale.setter
    def multi_scale(self, value):
        self._multi_scale = value

    def _to_channel_last(self, features: Dict[str, ms.Tensor]) -> Dict[str, ms.Tensor]:
        """Convert features to channel-last format."""
        return OrderedDict({k: ops.transpose(v, (0, 2, 3, 1)) for k, v in features.items()})

    def construct(self,x:ms.Tensor) -> Dict[str,ms.Tensor]:
        """Forward pass."""
        # Convert input format if needed
        if self.channel_last:
            x = ops.transpose(x, (0, 3, 1, 2))  # (B, H, W, C) -> (B, C, H, W)
        
        # Adjust input channels
        x = self.adjustment_layer(x)

        # Extract multi-scale features
        features = OrderedDict()
        for  i,layer in enumerate(self.body):
            x = layer(x)
            features[i] = x
        
        # Convert output format if needed
        if self.channel_last:
            features = self._to_channel_last(features)

        return features   
    
class Backbone(BackboneBase):
    def __init__(self,
                 name: str,
                 weights: str = '',
                 norm_layer: str = None,
                 in_channels: int = 3,
                 multi_scale: int = 1,
                 **kwargs):
        """Backbone wrapper for MindSpore.

        Arguments:
            name: Model name (e.g., 'resnet50').
            weights: Pretrained weights path or name (e.g., 'imagenet').
            norm_layer: Normalization layer (e.g., 'BatchNorm2d').
            in_channels: Input channels.
            multi_scale: Number of feature scales to return.
        """
        self.name = name.lower()
        self.weights = weights



        # Create backbone model
        backbone = create_model(
            self.name,
            pretrained=False,
            **kwargs
        )

        # Initialize base class
        super().__init__(
            backbone=backbone,
            in_channels=in_channels,
            multi_scale=multi_scale,
            weights=False,
            **kwargs
        )

    def _load_weights(self, weights_path: str) -> OrderedDict:
        """Load weights from checkpoint."""
        return load_checkpoint(weights_path)

    @staticmethod
    def _get_norm_layer(name: str) -> nn.Cell:
        """Get normalization layer by name."""
        norm_layers = {
            'batchnorm2d': nn.BatchNorm2d,
            'layernorm': nn.LayerNorm
        }
        return norm_layers.get(name.lower(), nn.BatchNorm2d)
    
def build_resnet(config: Dict[str, Any]) -> Backbone:
    """Helper function to build ResNet backbone."""
    return Backbone(**config)