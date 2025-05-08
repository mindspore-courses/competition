from mindspore.nn import Cell, Embedding, GELU, Dense, Dropout,ReLU, LayerNorm
from mindspore import Tensor, ops
import numpy as np
import mindspore as ms
import math
import numpy as np
from mindspore.nn.learning_rate_schedule import LearningRateSchedule, PolynomialDecayLR, CosineDecayLR
from mindspore import Tensor

class Gene2VecPositionalEmbedding(Cell):
    def __init__(self):
        super().__init__()
        gene2vec_weight = np.load('./data/gene2vec_16906.npy')
        gene2vec_weight = np.concatenate((gene2vec_weight, np.zeros((1, gene2vec_weight.shape[1]))), axis=0)
        gene2vec_weight = Tensor(gene2vec_weight, dtype=ms.float32)
        # 这里根据一个数据初始化它，还是需要注意的，不知道ms值不支持。
        # [16907,200]
        self.emb = Embedding(vocab_size=16907, embedding_size=200, embedding_table=gene2vec_weight, dtype=ms.float32)
        # self.emb.embedding_table.requires_grad=False

    def construct(self, x):
        # -> [16906,]
        t = ops.arange(start=0, end=x.shape[1], dtype=ms.int32)
        return self.emb(t)