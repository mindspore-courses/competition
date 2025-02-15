import mindspore as ms
import math
import copy
import numpy as np
import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore.ops.functional as F
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
from mindspore.ops.primitive import constexpr
from mindspore.common.initializer import TruncatedNormal, initializer, Normal
from mindspore import context
from functools import partial
from layers import Gene2VecPositionalEmbedding

# helpers
def exists(val):
    return val is not None
def empty(tensor):
    return tensor.numel() == 0
def default(val, d):
    return val if exists(val) else d


def softmax_kernel(data, projection_matrix, is_query=False, normalize_data=True, eps=1e-4):
    """
    data:[Batch,Heads,Seq,Dim_head]
    projection_matrix:[m,Dim_head]

    """
    b, h, Seq,Dim_head= data.shape
    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.
    ratio = (projection_matrix.shape[0] ** -0.5)
    # W'*X
    data_dash = data_normalizer * P.MatMul(transpose_b=True)(P.Reshape()(data,(-1,Dim_head)), projection_matrix)
    data_dash = P.Reshape()(data_dash,(b,h,Seq,-1))
    # |X|^2/2
    diag_data = data ** 2
    diag_data = P.ReduceSum(keep_dims=True)(diag_data, -1)
    diag_data = (diag_data / 2.0) * (data_normalizer ** 2)
    #exp(W'x-|X|^2/2)
    if is_query:
        data_dash = ratio * (
            P.Exp()(data_dash - diag_data -
                    P.ReduceMax(keep_dims=True)(data_dash, -1)) + eps)
    else:
        data_dash = ratio * (
            P.Exp()(data_dash - diag_data - P.ReduceMax()(data_dash)) + eps)

    return data_dash



def orthogonal_matrix_chunk(cols, qr_uniform_q = False):
    unstructured_block = np.random.randn(cols, cols).astype(np.float32)
    q, r = np.linalg.qr(unstructured_block,  mode='reduced')
    # proposed by @Parskatt
    # to make sure Q is uniform https://arxiv.org/pdf/math-ph/0609050.pdf
    if qr_uniform_q:
        d = np.diag(r, 0)
        q *= np.sign(d)
    # 转mindspore Tensor
    q = np.transpose(q)
    q = Tensor(q)
    return q

def gaussian_orthogonal_random_matrix(nb_rows, nb_columns, scaling = 0, qr_uniform_q = False):
    # print(nb_rows, nb_columns, scaling, qr_uniform_q)
    nb_full_blocks = int(nb_rows / nb_columns)
    block_list = []
    for _ in range(nb_full_blocks):
        q = orthogonal_matrix_chunk(nb_columns, qr_uniform_q = qr_uniform_q, )
        block_list.append(q)
    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        q = orthogonal_matrix_chunk(nb_columns, qr_uniform_q = qr_uniform_q, )
        block_list.append(q[:remaining_rows])
    final_matrix = P.Concat()(tuple(block_list))

    if scaling == 0:
        multiplier = Tensor(np.diag(np.linalg.norm(np.random.randn(nb_rows, nb_columns).astype(np.float32), axis = 1)))
    elif scaling == 1:
        multiplier = Tensor(np.diag(math.sqrt((float(nb_columns))) * np.ones((nb_rows,))))
    else:
        raise ValueError(f'Invalid scaling {scaling}')

    return P.MatMul()(multiplier,final_matrix)

class Softmax_kernel(nn.Cell):
    def __init__(self):
        super().__init__()
        self.Reshape = P.Reshape()
        self.MatMul_b = P.MatMul(transpose_b=True)
        self.ReduceSum = P.ReduceSum(keep_dims=True)
        self.Exp = P.Exp()
        self.ReduceMax_keep = P.ReduceMax(keep_dims=True)
        self.ReduceMax = P.ReduceMax()
    def construct(self, data, projection_matrix, is_query=False, normalize_data=True, eps=1e-4):
        """
        data:[Batch,Heads,Seq,Dim_head]
        projection_matrix:[m,Dim_head]

        """
        b, h, Seq, Dim_head = data.shape
        data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.
        ratio = (projection_matrix.shape[0] ** -0.5)
        # W'*X
        data_dash = data_normalizer * self.MatMul_b(self.Reshape(data, (-1, Dim_head)), projection_matrix)
        data_dash = self.Reshape(data_dash, (b, h, Seq, -1))
        # |X|^2/2
        diag_data = data ** 2
        diag_data = self.ReduceMax_keep(diag_data, -1)
        diag_data = (diag_data / 2.0) * (data_normalizer ** 2)
        # exp(W'x-|X|^2/2)
        if is_query:
            data_dash = ratio * (
                    self.Exp(data_dash - diag_data -
                            self.ReduceMax_keep(data_dash, -1)) + eps)
        else:
            data_dash = ratio * (
                    self.Exp(data_dash - diag_data - self.ReduceMax(data_dash)) + eps)

        return data_dash

class Linear_attention(nn.Cell):
    def __init__(self):
        super().__init__()
        self.ReduceSum =P.ReduceSum(keep_dims=True)
        self.BatchMatMul_b = P.BatchMatMul(transpose_b=True)
        self.BatchMatMul_a = P.BatchMatMul(transpose_a=True)
        self.BatchMatMul = P.BatchMatMul()
        self.Mul = P.Mul()
    def construct(self, q, k, v):
        """
        k,q,v:[B,Sq,H]
        """
        # [B,1,H]
        k_cumsum = self.ReduceSum(k, -2)
        # [B,Sq,1]
        D_inv = 1. /self.BatchMatMul_b(q, k_cumsum)
        # [B,H,H]
        context = self.BatchMatMul_a(k, v)
        # [B,Sq,H]
        out = self.BatchMatMul(q, context)
        # [B,Sq,H]*[B,Sq,1] ->
        out = self.Mul(out, D_inv)
        return out


class Causal_linear_attention(nn.Cell):
    def __init__(self):
        super().__init__()
        self.view_ = P.Reshape()
        self.CumSum = P.CumSum()
        self.ReduceSum =P.ReduceSum(keep_dims=True)
        self.BatchMatMul_b = P.BatchMatMul(transpose_b=True)
        self.BatchMatMul_a = P.BatchMatMul(transpose_a=True)
        self.Mul = P.Mul()
    def construct(self, q, k, v):
        k_cumsum = self.CumSum(k, -2)
        # [n,]
        D_inv = 1. / self.ReduceSum(q * k_cumsum, -1)
        # [n,d,1]*[n,1,e] -> [n,d,e]
        context = self.BatchMatMul_b(self.view_(k, k.shape + (1,)), self.view_(v, v.shape + (1,)))
        #[n,d,e] ->
        context = self.CumSum(context,-3)
        # [n,1,d] * [n,d,e] -> [n,1,e] = [n,e]
        out = self.BatchMatMul_a(self.view_(q, q.shape + (1,)), context)
        out = self.view_(out, v.shape)
        out = self.Mul(out, D_inv)
        return out

class LayerNorm(nn.Cell):
    """
    Layer Normalization

    Args:
        normalized_shape: the corresponding shape of the normalized axes
        eps: epsilon, a small number avoiding zero division

    Inputs:
        x: input tensor

    Returns:
        rescaled_output: Tensor, returned tensor after layernorm
    """
    def __init__(self, normalized_shape, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.gamma = Parameter(initializer('ones', normalized_shape), name="gamma")
        self.beta = Parameter(initializer('zeros', normalized_shape), name="beta")
        self.mean = P.ReduceMean(keep_dims=True)
        self.eps = eps

    def construct(self, x):
        mean = self.mean(x, -1)
        variance = self.mean(F.square(x - mean), -1)
        output = (x - mean) / F.sqrt(variance + self.eps)
        rescaled_output = output * self.gamma + self.beta
        return rescaled_output

class FeedForward(nn.Cell):
    def __init__(self, dim,
                 mult = 4,
                 initializer_range=0.02,
                 hidden_dropout_prob=0.1,
                 compute_type=mstype.float32):
        super(FeedForward,self).__init__()
        self.hidden_size = dim
        self.w1 = Mapping(dim,dim*mult,initializer_range,compute_type)
        self.w2 = Mapping(dim * mult,dim,initializer_range,compute_type)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(hidden_dropout_prob)
    def construct(self, x):
        x = self.w1(x)
        x = self.act(x)
        x = self.w2(x)
        x = self.dropout(x)
        return x

class Mapping(nn.Cell):
    """
    A mapping function with a 3d input
    Args:
        input_size: the size of the last dimension of the input tensor
        output_size: the desired size of the last dimension of the output tensor
        dtype: the compute datatype
        scale: the scale factor for initialization
    Inputs:
        x: the 3d input
    Returns:
        output: Tensor, a 3d tensor after projection
    """
    def __init__(self, input_size, output_size,initializer_range=0.02, dtype=ms.float32, scale=1.0):
        super(Mapping, self).__init__()
        self.output_size = output_size
        self.input_size = input_size
        self.weight = Parameter(initializer(Normal(sigma=initializer_range*scale), [input_size, output_size]),name="Weight")
        self.bias = Parameter(initializer("zeros", [output_size,]),name="Bias")
        self.dtype = dtype
        self.cast = P.Cast()

    def construct(self, x):
        out_shape = P.Shape()(x)[:-1] + (self.output_size,)
        x = P.Reshape()(x, (-1, self.input_size))
        x = nn.MatMul()(x, self.cast(self.weight, self.dtype)) + self.cast(self.bias, self.dtype)
        output = P.Reshape()(x, out_shape)
        return output

class FastAttention(nn.Cell):
    def __init__(self, dim_heads, nb_features = None, ortho_scaling = 0, causal = False, qr_uniform_q = False):
        super(FastAttention, self).__init__()
        nb_features = default(nb_features, int(dim_heads * math.log(dim_heads)))
        self.dim_heads = dim_heads
        self.nb_features = nb_features
        self.ortho_scaling = ortho_scaling
        ## projection_matrix is buffer
        self.projection_matrix = gaussian_orthogonal_random_matrix(nb_rows=self.nb_features,
                                                                   nb_columns=dim_heads,
                                                                   scaling=ortho_scaling,
                                                                   qr_uniform_q=qr_uniform_q)
        self.causal = causal
        self.attn_fn = Linear_attention() if not self.causal else Causal_linear_attention()
        self.softmax_kernel = Softmax_kernel()
    def construct(self, q, k, v):
        q = self.softmax_kernel(data=q, projection_matrix=self.projection_matrix, is_query=True)
        k = self.softmax_kernel(data=k, projection_matrix=self.projection_matrix, is_query=False)
        out = self.attn_fn(q, k, v)
        return out

class SelfAttention(nn.Cell):
    def __init__(self, dim, heads, dim_head, causal=False, nb_features=None, qr_uniform_q = False, dropout = 0.9):
        super(SelfAttention,self).__init__()
        assert dim % heads == 0, 'dimension must be divisible by number of heads'
        self.dim_head = dim_head
        self.fast_attention = FastAttention(dim_heads=self.dim_head, nb_features=nb_features, causal=causal, qr_uniform_q=qr_uniform_q)
        self.heads = heads
        self.to_q = Mapping(dim, dim)
        self.to_k = Mapping(dim, dim)
        self.to_v = Mapping(dim, dim)
        self.to_out = Mapping(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.view = P.Reshape()
        self.Concat = P.Concat(axis=1)
        self.Mul = P.Mul()
        self.ExpandDims = P.ExpandDims()
        self.Tile = P.Tile()
    def construct(self, x):
        """
        #b:batch_size
        #h:num_heads
        #n:seq_len
        #d:dim_perhead
        """
        b, n, dim, = x.shape
        h = self.heads

        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        q, k, v = self.view(q, (b,h,n,self.dim_head)), self.view(k, (b,h,n,self.dim_head)), self.view(v, (b,h,n,self.dim_head))

        # mask = self.Tile(self.ExpandDims(mask, -1), (1,1,dim))

        # v = self.Mul(v,self.view(mask,v.shape))

        out = self.fast_attention(q, k, v)
        out = self.view(out, (b,n,h* self.dim_head))
        out =  self.to_out(out)

        return self.dropout(out)

class EmbeddingLookup(nn.Cell):
    """
    A embeddings lookup table with a fixed dictionary and size.

    Args:
        vocab_size (int): Size of the dictionary of embeddings.
        embedding_size (int): The size of each embedding vector.
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form. Default: False.
        initializer_range (float): Initialization value of TruncatedNormal. Default: 0.02.
    """
    def __init__(self,
                 vocab_size,
                 embedding_size,
                 use_one_hot_embeddings=False,
                 initializer_range=0.02):
        super(EmbeddingLookup, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.use_one_hot_embeddings = use_one_hot_embeddings
        self.embedding_table = Parameter(initializer(Normal(sigma=initializer_range),
                                                     [vocab_size, embedding_size]), name="embedding_table")
        self.expand = P.ExpandDims()
        self.shape_flat = (-1,)
        self.gather = P.GatherV2()
        self.one_hot = P.OneHot()
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.array_mul = P.MatMul()
        self.reshape = P.Reshape()
        self.shape = P.Shape()

    def construct(self, input_ids):
        """Get a embeddings lookup table with a fixed dictionary and size."""
        input_shape = self.shape(input_ids)

        flat_ids = self.reshape(input_ids, self.shape_flat)
        if self.use_one_hot_embeddings:
            one_hot_ids = self.one_hot(flat_ids, self.vocab_size, self.on_value, self.off_value)
            output_for_reshape = self.array_mul(one_hot_ids, self.embedding_table)
        else:
            output_for_reshape = self.gather(self.embedding_table, flat_ids, 0)

        out_shape = input_shape + (self.embedding_size,)
        output = self.reshape(output_for_reshape, out_shape)
        return output

class AbsolutePositionalEmbedding(nn.Cell):
    def __init__(self, dim, max_seq_len):
        super(AbsolutePositionalEmbedding, self).__init__()
        self.emb = nn.EmbeddingLookup(max_seq_len, dim)

    def construct(self, x):
        batch_size, seq_length = x.shape[0], x.shape[1]
        input_position = F.tuple_to_array(F.make_range(seq_length))
        # input_position = P.Tile()(input_position, (batch_size, 1))
        return self.emb(input_position)


class Performer_layer(nn.Cell):
    def __init__(self,dim, heads, dim_head, causal=False, nb_features=None, qr_uniform_q = False, dropout = 0.9):
        super(Performer_layer, self).__init__()
        self.SelfAttention = SelfAttention(dim, heads, dim_head, causal, nb_features, qr_uniform_q, dropout)
        self.FeedForward = FeedForward(dim=dim)
        self.LayerNorm = LayerNorm(dim,)
    def construct(self, x):

        x = self.LayerNorm(x)
        out = x + self.SelfAttention(x)
        out = self.LayerNorm(out)
        out = out + self.FeedForward(x)
        return out

class Performer(nn.Cell):
    def __init__(self,dim, depth, heads, causal=False, nb_features=None, qr_uniform_q = False, dropout = 0.9):
        super(Performer, self).__init__()
        assert dim % heads == 0
        dim_head = dim//heads
        layers = []
        for _ in range(depth):
            layers.append(Performer_layer(dim=dim, heads=heads,
                                          dim_head=dim_head,
                                          causal=causal,
                                          nb_features=nb_features,
                                          qr_uniform_q=qr_uniform_q,
                                          dropout=dropout ))

        self.layers = nn.CellList(layers)

    def construct(self, input_tensor):
        prev_output = input_tensor
        for layer_module in self.layers:
            prev_output = layer_module(prev_output)
        return prev_output

class PerformerLM(nn.Cell):
    def __init__(self, num_tokens, max_seq_len, dim, depth, heads, causal = True,
                 nb_features = None, emb_dropout = 0.9, pf_dropout = 0.9, qr_uniform_q = False):
        super(PerformerLM,self).__init__()
        self.max_seq_len = max_seq_len
        self.dim = dim
        self.num_tokens = num_tokens
        self.token_emb = EmbeddingLookup(num_tokens, dim)
        # self.pos_emb = AbsolutePositionalEmbedding(dim, max_seq_len)
        self.pos_emb = Gene2VecPositionalEmbedding()
        self.dropout = nn.Dropout(emb_dropout)
        self.performer = Performer(dim, depth, heads, causal, nb_features, qr_uniform_q, pf_dropout )
        self.norm = LayerNorm(dim)
        self.MatMul = P.MatMul(transpose_b=True)
        self.Reshape = P.Reshape()
        self.to_out = nn.Dense(dim, num_tokens, dtype=ms.float32)
    def construct(self, input_ids):
        # b, n = input_ids.shape
        # assert n <= self.max_seq_len, f'sequence length {n} must be less than the max sequence length {self.max_seq_len}'
        # token and positional embeddings
        
        x = self.token_emb(input_ids)
        x += self.pos_emb(x)
        x = self.dropout(x)
        x = self.performer(x)
        # norm and to logits
        #[batch,seq,hidden]
        x = self.norm(x)
        # res = self.MatMul(self.Reshape(x,(-1,self.dim)), self.token_emb.embedding_table)
        # return self.Reshape(res, input_ids.shape+(self.num_tokens,))
        # 5. (batch, 16906, 200) -> (batch, 16906, 7)
        # 输出层
        x = self.to_out(x)
        return x