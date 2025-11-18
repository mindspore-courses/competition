import mindspore
from mindspore import Tensor
import numpy as np
import scanpy as sc
from sklearn.model_selection import train_test_split

# 预训练用的不带标签的
class SCDataset:
    def __init__(self, data, n_class, seq_len):
        self.data = data
        self.n_class = n_class
        self.seq_len = seq_len

    def __getitem__(self, index):
        full_seq = self.data[index].toarray()[0]  # 假设输入data是稀疏矩阵格式
        full_seq[full_seq > (self.n_class - 2)] = self.n_class - 2
        full_seq = np.append(full_seq, 0).astype(np.int32)  # 添加额外的类别
        # print(full_seq)
        return Tensor(full_seq[:self.seq_len])

    def __len__(self):
        return self.data.shape[0]

    # MindSpore特定: 转换为MindSpore数据集
    def to_mind_dataset(self, batch_size=32, repeat_size=1):
        def generator():
            for i in range(len(self)):
                yield self[i],
        
        # 创建数据集
        types = [mindspore.int32,]
        ds = mindspore.dataset.GeneratorDataset(generator, column_names=["data"], column_types=types)
        ds = ds.batch(batch_size).repeat(repeat_size)
        return ds
    
def load_data(data_path, n_class, seed, batch_size, seq_len):
    data = sc.read_h5ad(data_path, backed="r")
    data = data.X[:100]
    data_train, data_val = train_test_split(data, test_size=0.05,random_state=seed)
    train_dataset = SCDataset(data_train, n_class, seq_len).to_mind_dataset(batch_size=batch_size)
    val_dataset = SCDataset(data_val, n_class, seq_len).to_mind_dataset(batch_size=batch_size)
    print("load data success, train num is {}, val num is {}".format(len(train_dataset), len(val_dataset)))
    return train_dataset, val_dataset
