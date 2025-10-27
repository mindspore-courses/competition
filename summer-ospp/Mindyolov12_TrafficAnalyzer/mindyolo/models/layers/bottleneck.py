import mindspore as ms
from mindspore import nn, ops, Parameter, Tensor
from mindspore.common.initializer import initializer, TruncatedNormal, Constant

from .activation import SiLU
from .conv import ConvNormAct, DWConvNormAct, RepConv
from .flash_attn_interface import flash_attn_func


class Bottleneck(nn.Cell):
    # Standard bottleneck
    def __init__(
        self, c1, c2, shortcut=True, k=(1, 3), g=(1, 1), e=0.5, act=True, momentum=0.97, eps=1e-3, sync_bn=False
    ):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.conv1 = ConvNormAct(c1, c_, k[0], 1, g=g[0], act=act, momentum=momentum, eps=eps, sync_bn=sync_bn)
        self.conv2 = ConvNormAct(c_, c2, k[1], 1, g=g[1], act=act, momentum=momentum, eps=eps, sync_bn=sync_bn)
        self.add = shortcut and c1 == c2

    def construct(self, x):
        if self.add:
            out = x + self.conv2(self.conv1(x))
        else:
            out = self.conv2(self.conv1(x))
        return out

    
class Residualblock(nn.Cell):
    def __init__(
        self, c1, c2, k=(1, 3), g=(1, 1), act=True, momentum=0.97, eps=1e-3, sync_bn=False
    ):  # ch_in, ch_out, kernels, groups, expand
        super().__init__()
        self.conv1 = ConvNormAct(c1, c2, k[0], 1, g=g[0], act=act, momentum=momentum, eps=eps, sync_bn=sync_bn)
        self.conv2 = ConvNormAct(c2, c2, k[1], 1, g=g[1], act=act, momentum=momentum, eps=eps, sync_bn=sync_bn)

    def construct(self, x):
        out = x + self.conv2(self.conv1(x))
        return out


class C3(nn.Cell):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, momentum=0.97, eps=1e-3, sync_bn=False):
        super(C3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.conv1 = ConvNormAct(c1, c_, 1, 1, momentum=momentum, eps=eps, sync_bn=sync_bn)
        self.conv2 = ConvNormAct(c1, c_, 1, 1, momentum=momentum, eps=eps, sync_bn=sync_bn)
        self.conv3 = ConvNormAct(2 * c_, c2, 1, momentum=momentum, eps=eps, sync_bn=sync_bn)  # act=FReLU(c2)
        self.m = nn.SequentialCell(
            [
                Bottleneck(c_, c_, shortcut, k=(1, 3), g=(1, g), e=1.0, momentum=momentum, eps=eps, sync_bn=sync_bn)
                for _ in range(n)
            ]
        )
        self.concat = ops.Concat(axis=1)

    def construct(self, x):
        c1 = self.conv1(x)
        c2 = self.m(c1)
        c3 = self.conv2(x)
        c4 = self.concat((c2, c3))
        c5 = self.conv3(c4)

        return c5


class C2f(nn.Cell):
    # CSP Bottleneck with 2 convolutions
    def __init__(
        self, c1, c2, n=1, shortcut=False, g=1, e=0.5, momentum=0.97, eps=1e-3, sync_bn=False
    ):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = ConvNormAct(c1, 2 * self.c, 1, 1, momentum=momentum, eps=eps, sync_bn=sync_bn)
        self.cv2 = ConvNormAct(
            (2 + n) * self.c, c2, 1, momentum=momentum, eps=eps, sync_bn=sync_bn
        )  # optional act=FReLU(c2)
        self.m = nn.CellList(
            [
                Bottleneck(self.c, self.c, shortcut, k=(3, 3), g=(1, g), e=1.0, momentum=momentum, eps=eps, sync_bn=sync_bn)
                for _ in range(n)
            ]
        )

    def construct(self, x):
        y = ()
        x = self.cv1(x)
        _c = x.shape[1] // 2
        x_tuple = ops.split(x, axis=1, split_size_or_sections=_c)
        y += x_tuple
        for i in range(len(self.m)):
            m = self.m[i]
            out = m(y[-1])
            y += (out,)

        return self.cv2(ops.concat(y, axis=1))


class DWBottleneck(nn.Cell):
    # depthwise bottleneck used in yolox nano scale
    def __init__(
        self, c1, c2, shortcut=True, k=(1, 3), e=0.5, act=True, momentum=0.97, eps=1e-3, sync_bn=False
    ):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.conv1 = ConvNormAct(c1, c_, k[0], 1, act=True, momentum=momentum, eps=eps, sync_bn=sync_bn)
        self.conv2 = DWConvNormAct(c_, c2, k[1], 1, act=True, momentum=momentum, eps=eps, sync_bn=sync_bn)
        self.add = shortcut and c1 == c2

    def construct(self, x):
        if self.add:
            out = x + self.conv2(self.conv1(x))
        else:
            out = self.conv2(self.conv1(x))
        return out


class DWC3(nn.Cell):
    # depthwise DwC3 used in yolox nano scale, similar as C3
    def __init__(self, c1, c2, n=1, shortcut=True, e=0.5, momentum=0.97, eps=1e-3, sync_bn=False):
        super(DWC3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.conv1 = ConvNormAct(c1, c_, 1, 1, momentum=momentum, eps=eps, sync_bn=sync_bn)
        self.conv2 = ConvNormAct(c1, c_, 1, 1, momentum=momentum, eps=eps, sync_bn=sync_bn)
        self.conv3 = ConvNormAct(2 * c_, c2, 1, momentum=momentum, eps=eps, sync_bn=sync_bn)  # act=FReLU(c2)
        self.m = nn.SequentialCell(
            [
                DWBottleneck(c_, c_, shortcut, k=(1, 3), e=1.0, momentum=momentum, eps=eps, sync_bn=sync_bn)
                for _ in range(n)
            ]
        )
        self.concat = ops.Concat(axis=1)

    def construct(self, x):
        c1 = self.conv1(x)
        c2 = self.m(c1)
        c3 = self.conv2(x)
        c4 = self.concat((c2, c3))
        c5 = self.conv3(c4)

        return c5


class RepNBottleneck(nn.Cell):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5, sync_bn=False):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConv(c1, c_, k[0], 1, bn=False, sync_bn=sync_bn)
        self.cv2 = ConvNormAct(c_, c2, k[1], 1, g=g, sync_bn=sync_bn)
        self.add = shortcut and c1 == c2

    def construct(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class RepNCSP(nn.Cell):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, sync_bn=False):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(RepNCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = ConvNormAct(c1, c_, 1, 1, sync_bn=sync_bn)
        self.cv2 = ConvNormAct(c1, c_, 1, 1, sync_bn=sync_bn)
        self.cv3 = ConvNormAct(2 * c_, c2, 1, sync_bn=sync_bn)  # optional act=FReLU(c2)
        self.m = nn.SequentialCell(*(RepNBottleneck(c_, c_, shortcut, g, e=1.0, sync_bn=sync_bn) for _ in range(n)))

    def construct(self, x):
        return self.cv3(ops.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class RepNCSPELAN4(nn.Cell):
    # csp-elan
    def __init__(self, c1, c2, c3, c4, c5=1, sync_bn=False):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(RepNCSPELAN4, self).__init__()
        self.c = c3//2
        self.cv1 = ConvNormAct(c1, c3, 1, 1, sync_bn=sync_bn)
        self.cv2 = nn.SequentialCell(RepNCSP(c3//2, c4, c5, sync_bn=sync_bn), ConvNormAct(c4, c4, 3, 1, sync_bn=sync_bn))
        self.cv3 = nn.SequentialCell(RepNCSP(c4, c4, c5, sync_bn=sync_bn), ConvNormAct(c4, c4, 3, 1, sync_bn=sync_bn))
        self.cv4 = ConvNormAct(c3+(2*c4), c2, 1, 1, sync_bn=sync_bn)

    def construct(self, x):
        y = ()
        x = self.cv1(x)
        _c = x.shape[1] // 2
        x_tuple = ops.split(x, axis=1, split_size_or_sections=_c)
        y += x_tuple
        for m in [self.cv2, self.cv3]:
            out = m(y[-1])
            y += (out,)
        return self.cv4(ops.cat(y, 1))


class SCDown(nn.Cell):
    def __init__(self, c1, c2, k, s):
        super().__init__()
        self.cv1 = ConvNormAct(c1, c2, k=1, s=1)
        self.cv2 = ConvNormAct(c2, c2, k=k, s=s, g=c2, act=False)
    
    def construct(self, x):
        return self.cv2(self.cv1(x))


class Attention(nn.Cell):
    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = ConvNormAct(c1=dim, c2=h, k=1, act=False)
        self.proj = ConvNormAct(c1=dim, c2=dim, k=1, act=False)
        self.pe = ConvNormAct(c1=dim, c2=dim, k=3, s=1, g=dim, act=False)

    def construct(self, x):
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim*2 + self.head_dim, N).split([self.key_dim, self.key_dim, self.head_dim], axis=2)
        
        attn = (
            (ops.transpose(q, (0, 1, 3, 2)) @ k) * self.scale
        )
        attn = ops.softmax(attn)
        x = (v @ ops.transpose(attn, (0, 1, 3, 2))).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x


class PSA(nn.Cell):
    def __init__(self, c1, c2, e=0.5):
        super().__init__()
        assert(c1 == c2)
        self.c = int(c1 * e)
        self.cv1 = ConvNormAct(c1, 2 * self.c, 1, 1)
        self.cv2 = ConvNormAct(2 * self.c, c1, 1)

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c // 64)
        self.ffn = nn.SequentialCell(
            [
                ConvNormAct(self.c, self.c*2, 1),
                ConvNormAct(self.c*2, self.c, 1, act=False)
            ]
        )
    
    def construct(self, x):
        a, b = self.cv1(x).split((self.c, self.c), axis=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(ops.concat((a, b), 1))


class RepVGGDW(nn.Cell):
    def __init__(self, ed):
        super().__init__()
        self.conv = ConvNormAct(ed, ed, k=7, s=1, p=3, g=ed, act=False)
        self.conv1 = ConvNormAct(ed, ed, k=3, s=1, p=1, g=ed, act=False)
        self.dim = ed
        self.act = SiLU()
    
    def construct(self, x):
        return self.act(self.conv(x) + self.conv1(x))


class CIB(nn.Cell):
    # Standard bottleneck
    def __init__(
        self, c1, c2, shortcut=True, e=0.5, lk=False, act=True, momentum=0.97, eps=1e-3, sync_bn=False
    ):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.SequentialCell(
            [
                ConvNormAct(c1, c1, 3, g=c1, act=act, momentum=momentum, eps=eps, sync_bn=sync_bn),
                ConvNormAct(c1, 2 * c_, 1, act=act, momentum=momentum, eps=eps, sync_bn=sync_bn),
                ConvNormAct(2 * c_, 2 * c_, 3, g=2 * c_, act=act, momentum=momentum, eps=eps, sync_bn=sync_bn) if not lk else RepVGGDW(2 * c_),
                ConvNormAct(2 * c_, c2, 1, act=act, momentum=momentum, eps=eps, sync_bn=sync_bn),
                ConvNormAct(c2, c2, 3, g=c2, act=act, momentum=momentum, eps=eps, sync_bn=sync_bn),
            ]
        )
        self.add = shortcut and c1 == c2
    
    def construct(self, x):
        if self.add:
            out = x + self.cv1(x)
        else:
            out = self.cv1(x)
        return out


class C2fCIB(C2f):
    # CSP Bottleneck with 2 convolutions
    def __init__(
        self, c1, c2, n=1, shortcut=False, lk=False, g=1, e=0.5, momentum=0.97, eps=1e-3, sync_bn=False
    ):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e, momentum, eps, sync_bn)
        self.m = nn.CellList(
            [
                CIB(self.c, self.c, shortcut, e=1.0, lk=lk, momentum=momentum, eps=eps, sync_bn=sync_bn)
                for _ in range(n)
            ]
        )


class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True, sync_bn=False):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e, sync_bn=sync_bn)
        self.m = nn.CellList(
            [C3k(self.c, self.c, 2, shortcut, g, sync_bn=sync_bn) if c3k else Bottleneck(self.c, self.c, shortcut, k=(3, 3), g=(1,g), sync_bn=sync_bn) for _ in range(n)]
        )


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3, sync_bn=False):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e, sync_bn=sync_bn)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.SequentialCell(*(Bottleneck(c_, c_, shortcut, k=(k, k), g=(1, g), e=1.0, sync_bn=sync_bn) for _ in range(n)))


class PSABlock(nn.Cell):
    """
    PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = mindspore.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True, sync_bn=False) -> None:
        """Initializes the PSABlock with attention and feed-forward layers for enhanced feature extraction."""
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.SequentialCell(ConvNormAct(c, c * 2, 1, sync_bn=sync_bn), ConvNormAct(c * 2, c, 1, act=False, sync_bn=sync_bn))
        self.add = shortcut

    def construct(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class C2PSA(nn.Cell):
    """
    C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = mindspore.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)
    """

    def __init__(self, c1, c2, n=1, e=0.5, sync_bn=False):
        """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = ConvNormAct(c1, 2 * self.c, 1, 1, sync_bn=sync_bn)
        self.cv2 = ConvNormAct(2 * self.c, c1, 1, sync_bn=sync_bn)

        self.m = nn.SequentialCell(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64, sync_bn=sync_bn) for _ in range(n)))

    def construct(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        x = self.cv1(x)
        a, b = ops.split(x, axis=1, split_size_or_sections=self.c)
        b = self.m(b)
        return self.cv2(ops.cat((a, b), 1))
    
class AAttn(nn.Cell):
    """
    Area-attention module with the requirement of flash attention.

    Attributes:
        dim (int): Number of hidden channels;
        num_heads (int): Number of heads into which the attention mechanism is divided;
        area (int, optional): Number of areas the feature map is divided. Defaults to 1.
        sync_bn (bool, optional): Whether to use synchronized batch normalization. Defaults to False.

    Methods:
        construct: Performs a forward process of input tensor and outputs a tensor after the execution of the area attention mechanism.

    Examples:
        >>> import mindspore
        >>> from mindyolo.models.layers import AAttn
        >>> model = AAttn(dim=64, num_heads=2, area=4)
        >>> x = mindspore.ops.randn(2, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    
    Notes: 
        recommend that dim//num_heads be a multiple of 32 or 64.
    """

    def __init__(self, dim, num_heads, area=1, sync_bn=False):
        """Initializes the area-attention module, a simple yet efficient attention module for YOLO."""
        super().__init__()
        self.area = area
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads

        # Query and Key projection
        self.qk = ConvNormAct(dim, all_head_dim * 2, 1, act=False, sync_bn=sync_bn)
        # Value projection
        self.v = ConvNormAct(dim, all_head_dim, 1, act=False, sync_bn=sync_bn)
        # Output projection
        self.proj = ConvNormAct(all_head_dim, dim, 1, act=False, sync_bn=sync_bn)
        
        # Positional encoding
        self.pe = ConvNormAct(all_head_dim, dim, 5, s=1, p=2, g=dim, act=False, sync_bn=sync_bn)
        self.USE_FLASH_ATTN = True

    def construct(self, x):
        """Processes the input tensor 'x' through the area-attention"""
        B, C, H, W = x.shape
        N = H * W

        # Generate query, key pairs
        # qk = self.qk(x).view(B, -1, N).transpose(0, 2, 1)  # B, N, 2*C
        qk = self.qk(x).flatten(start_dim=2).transpose(0, 2, 1)
        
        # Generate value
        v = self.v(x)
        pp = self.pe(v)  # Positional encoding
        # v = v.view(B, -1, N).transpose(0, 2, 1)  # B, N, C
        v = v.flatten(start_dim=2).transpose(0, 2, 1)

        # Area-based attention
        if self.area > 1:
            # Reshape for area-based processing
            qk = qk.view(B * self.area, N // self.area, C * 2)
            v = v.view(B * self.area, N // self.area, C)
            B, N, _ = qk.shape   
        q, k = ops.split(qk, axis=2, split_size_or_sections=C)

        if self.USE_FLASH_ATTN:
            q = q.view(B, N, self.num_heads, self.head_dim)
            k = k.view(B, N, self.num_heads, self.head_dim)
            v = v.view(B, N, self.num_heads, self.head_dim)

            x = flash_attn_func(
            q,
            k,
            v,
            dropout_p=0.0,
            softmax_scale=None,
            causal=False)
            
        else:
            # Standard attention without area division
            q, k = ops.split(qk, axis=2, split_size_or_sections=C)
            
            # Reshape for multi-head attention
            q = q.transpose(0, 2, 1).view(B, self.num_heads, self.head_dim, N)
            k = k.transpose(0, 2, 1).view(B, self.num_heads, self.head_dim, N)
            v = v.transpose(0, 2, 1).view(B, self.num_heads, self.head_dim, N)
            
            # Compute attention
            attn = ops.matmul(q.transpose(0, 1, 3, 2), k) * (self.head_dim ** -0.5)
            
            # 修复：使用ops.ReduceMax替代ops.max，确保数据类型一致
            reduce_max = ops.ReduceMax(keep_dims=True)
            max_attn = reduce_max(attn, -1)
            exp_attn = ops.exp(attn - max_attn)

            reduce_sum = ops.ReduceSum(keep_dims=True)
            sum_attn = reduce_sum(exp_attn, -1)
            attn = exp_attn / sum_attn

            # Apply attention
            # x = ops.matmul(v, attn.transpose(-2, -1))
            x = ops.matmul(v, attn.transpose(0, 1, 3, 2))
            x = x.transpose(0, 3, 1, 2)  # B, N, num_heads, head_dim
        
        if self.area > 1:
            x = x.reshape(B // self.area, N * self.area, C)
            B, N, _ = x.shape

        # Reshape to spatial format
        x = x.view(B, H, W, C).transpose(0, 3, 1, 2)  # B, C, H, W
        
        # Apply output projection with positional encoding
        return self.proj(x + pp)
    
class ABlock(nn.Cell):
    """
    ABlock class implementing a Area-Attention block with effective feature extraction.

    This class encapsulates the functionality for applying multi-head attention with feature map are dividing into areas
    and feed-forward neural network layers.

    Attributes:
        dim (int): Number of hidden channels;
        num_heads (int): Number of heads into which the attention mechanism is divided;
        mlp_ratio (float, optional): MLP expansion ratio (or MLP hidden dimension ratio). Defaults to 1.2;
        area (int, optional): Number of areas the feature map is divided.  Defaults to 1.
        sync_bn (bool, optional): Whether to use synchronized batch normalization. Defaults to False.

    Methods:
        construct: Performs a forward pass through the ABlock, applying area-attention and feed-forward layers.

    Examples:
        Create a ABlock and perform a forward pass
        >>> model = ABlock(dim=64, num_heads=2, mlp_ratio=1.2, area=4)
        >>> x = mindspore.ops.randn(2, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    
    Notes: 
        recommend that dim//num_heads be a multiple of 32 or 64.
    """

    def __init__(self, dim, num_heads, mlp_ratio=1.2, area=1, sync_bn=False):
        """Initializes the ABlock with area-attention and feed-forward layers for faster feature extraction."""
        super().__init__()

        self.attn = AAttn(dim, num_heads=num_heads, area=area, sync_bn=sync_bn)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.SequentialCell([
            ConvNormAct(dim, mlp_hidden_dim, 1, sync_bn=sync_bn), 
            ConvNormAct(mlp_hidden_dim, dim, 1, act=False, sync_bn=sync_bn)
        ])

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using a truncated normal distribution."""
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(initializer(TruncatedNormal(sigma=0.02), cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(initializer(Constant(0), cell.bias.shape, cell.bias.dtype))

    def apply(self, fn):
        for cell in self.cells():
            cell.apply(fn)
        fn(self)
        return self

    def construct(self, x):
        """Executes a forward pass through ABlock, applying area-attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x

class A2C2f(nn.Cell):
    """
    A2C2f module with residual enhanced feature extraction using ABlock blocks with area-attention. Also known as R-ELAN

    This class extends the C2f module by incorporating ABlock blocks for fast attention mechanisms and feature extraction.

    Attributes:
        c1 (int): Number of input channels;
        c2 (int): Number of output channels;
        n (int, optional): Number of 2xABlock modules to stack. Defaults to 1;
        a2 (bool, optional): Whether use area-attention. Defaults to True;
        area (int, optional): Number of areas the feature map is divided. Defaults to 1;
        residual (bool, optional): Whether use the residual (with layer scale). Defaults to False;
        mlp_ratio (float, optional): MLP expansion ratio (or MLP hidden dimension ratio). Defaults to 1.2;
        e (float, optional): Expansion ratio for R-ELAN modules. Defaults to 0.5;
        g (int, optional): Number of groups for grouped convolution. Defaults to 1;
        shortcut (bool, optional): Whether to use shortcut connection. Defaults to True;
        sync_bn (bool, optional): Whether to use synchronized batch normalization. Defaults to False;

    Methods:
        construct: Performs a forward pass through the A2C2f module.

    Examples:
        >>> import mindspore
        >>> from mindyolo.models.layers import A2C2f
        >>> model = A2C2f(c1=64, c2=64, n=2, a2=True, area=4, residual=True, e=0.5)
        >>> x = mindspore.ops.randn(2, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    """

    def __init__(self, c1, c2, n=1, a2=True, area=1, residual=False, mlp_ratio=2.0, e=0.5, g=1, shortcut=True, sync_bn=False):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        assert c_ % 32 == 0, "Dimension of ABlock be a multiple of 32."

        # num_heads = c_ // 64 if c_ // 64 >= 2 else c_ // 32
        num_heads = c_ // 32

        self.cv1 = ConvNormAct(c1, c_, 1, 1, sync_bn=sync_bn)
        self.cv2 = ConvNormAct((1 + n) * c_, c2, 1, sync_bn=sync_bn)  # optional act=FReLU(c2)

        # Layer scale parameter for residual connection
        init_values = 0.01  # or smaller
        if a2 and residual:
            self.gamma = Parameter(Tensor(init_values * ops.ones((c2,), ms.float32)), requires_grad=True)
        else:
            self.gamma = None

        # Build modules list
        self.m = nn.CellList()
        for _ in range(n):
            if a2:
                # Use ABlock modules with area attention
                ablock_seq = nn.SequentialCell([
                    ABlock(c_, num_heads, mlp_ratio, area, sync_bn=sync_bn) for _ in range(2)
                ])
                self.m.append(ablock_seq)
            else:
                # Use C3k module as fallback
                self.m.append(C3k(c_, c_, 2, shortcut, g, sync_bn=sync_bn))

    def construct(self, x):
        """Forward pass through R-ELAN layer."""
        y = [self.cv1(x)]
        
        # Process through each module
        for m in self.m:
            y.append(m(y[-1]))
        
        # Concatenate all outputs
        out = self.cv2(ops.concat(y, axis=1))
        
        # Apply residual connection with layer scale if enabled
        if self.gamma is not None:
            gamma_reshaped = self.gamma.view(1, -1, 1, 1)
            return x + gamma_reshaped * out
        
        return out