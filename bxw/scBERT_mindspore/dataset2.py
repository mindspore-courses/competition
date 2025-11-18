import mindspore
from mindspore import Tensor
import numpy as np
import scanpy as sc
from sklearn.model_selection import train_test_split

# 微调用的带有标签的类型
class SCDataset:
    def __init__(self, data, labels, n_class):
        self.data = data
        self.labels = labels
        self.n_class = n_class

    def __getitem__(self, index):
        full_seq = self.data[index].toarray()[0]  # 假设输入data是稀疏矩阵格式
        full_seq[full_seq > (self.n_class - 2)] = self.n_class - 2
        full_seq = np.append(full_seq, 0).astype(np.int32)  # 添加额外的类别
        label = self.labels[index]
        label = np.array(label, dtype=np.int32)
        return Tensor(full_seq), Tensor(label)

    def __len__(self):
        return self.data.shape[0]

    # MindSpore特定: 转换为MindSpore数据集
    def to_mind_dataset(self, batch_size=32, repeat_size=1):
        def generator():
            for i in range(len(self)):
                # yield self[i],
                data, label = self[i]  # 假设 self[i] 返回一个 (data, label) 元组
                yield (data, label)
        
        # 创建数据集
        types = [mindspore.int32, mindspore.int32]  
        c_names = ["data", "label"]  
        ds = mindspore.dataset.GeneratorDataset(generator, column_names=c_names, column_types=types)
        ds = ds.batch(batch_size).repeat(repeat_size)
        return ds

def load_data2(data_path, n_class, seed, batch_size):
    data = sc.read_h5ad(data_path)
    label_dict, label = np.unique(np.array(data.obs['celltype']), return_inverse=True)  # Convert strings categorical to integrate categorical, and label_dict[label] can be restored
    data = data.X
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.1, random_state=42)
    train_dataset = SCDataset(X_train, y_train, n_class).to_mind_dataset(batch_size=batch_size)
    val_dataset = SCDataset( X_test, y_test, n_class).to_mind_dataset(batch_size=batch_size)
    print("load data success, train num is {}, val num is {}".format(len(train_dataset), len(val_dataset)))
    return train_dataset, val_dataset