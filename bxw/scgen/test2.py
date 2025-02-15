import os
from random import shuffle
import anndata
import matplotlib
import numpy as np
import scanpy as sc
import sklearn as sk
import mindspore as ms
import mindspore.ops as ops
from mindspore import Tensor, dtype as mstype
from mindspore.train.serialization import save_checkpoint
import mindspore.nn as nn
from mindspore.train.serialization import load_checkpoint, load_param_into_net


ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend")

# 使用训练好的模型，进行数据的批次去除。
train_path = "/home/ma-user/work/scgen_tensorflow/data/pancreas.h5ad"

data = sc.read(train_path)

path_to_save = "/home/ma-user/work/scgen_tensorflow/results/Figure 6/"
sc.settings.figdir = path_to_save
model_to_use = "/home/ma-user/work/scgen_tensorflow/models/scGen/scgen.pt"
batch_size = 32
input_matrix = data.X
ind_list = [i for i in range(input_matrix.shape[0])]
shuffle(ind_list)
train_data = input_matrix[ind_list, :]
gex_size = input_matrix.shape[1]
X_dim = gex_size
z_dim = 100
lr = 0.001
dr_rate = 0.2

data_max_value = np.amax(input_matrix)


# Define the VAE model


class VAE(nn.Cell):
    def __init__(self, input_dim, hidden_dim=800, z_dim=100, dr_rate=0.2):
        super(VAE, self).__init__()

        # =============================== Q(z|X) ======================================
        self.encoder = nn.SequentialCell([
            nn.Dense(input_dim, hidden_dim, has_bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(alpha=0.01),
            nn.Dropout(p=dr_rate),
            nn.Dense(hidden_dim, hidden_dim, has_bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=dr_rate),
        ])
        self.fc_mean = nn.Dense(hidden_dim, z_dim)
        self.fc_var = nn.Dense(hidden_dim, z_dim)

        # =============================== P(X|z) ======================================
        self.decoder = nn.SequentialCell([
            nn.Dense(z_dim, hidden_dim, has_bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(alpha=0.01),
            nn.Dropout(p=dr_rate),
            nn.Dense(hidden_dim, hidden_dim, has_bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=dr_rate),
            nn.Dense(hidden_dim, input_dim),
            nn.ReLU(),
        ])

        self.exp = ops.Exp()
        self.randn_like = ops.StandardNormal()

    def encode(self, x):
        h = self.encoder(x)
        mean = self.fc_mean(h)
        log_var = self.fc_var(h)
        return mean, log_var

    def reparameterize(self, mu, log_var):
        # 计算标准差
        std = self.exp(0.5 * log_var)
        
        # 获取 std 的形状并转换为整数类型的 Tensor
        shape_tuple = ops.Shape()(std)  # 获取形状，返回一个 Python 元组
        shape = Tensor(list(shape_tuple), mstype.int32)  # 转换为 Tensor[int32]
        
        # 生成与 std 相同形状的标准正态分布噪声
        eps = self.randn_like(shape)
        
        # 重参数化
        return mu + eps * std



    def decode(self, z):
        x_hat = self.decoder(z)
        return x_hat

    def construct(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z)
        return x_hat, mu, log_var


# Initialize the model
model = VAE(input_dim=X_dim, z_dim=z_dim, dr_rate=dr_rate)
optimizer = ms.experimental.optim.Adam(model.trainable_params(), lr=lr)


def give_me_latent(model, data):
    """
    从给定的模型和数据中提取潜在表示（z_mean）。

    参数:
    - model: 已训练的 VAE 模型（继承自 mindspore.nn.Cell）
    - data: 输入数据，类型可以是 NumPy 数组或其他可转换为 Tensor 的格式

    返回:
    - z_mean: NumPy 数组形式的潜在表示
    """
    # 1. 设置模型为评估模式
    model.set_train(False)

    # 2. 将输入数据转换为 MindSpore Tensor
    #    MindSpore 会根据上下文自动将 Tensor 放置在正确的设备上（CPU/GPU）
    data_tensor = Tensor(data, dtype=ms.float32)

    # 3. 前向传播以获取 mu 和 log_var
    mu, log_var = model.encode(data_tensor)

    # 4. 获取潜在均值 z_mean
    z_mean = mu

    # 5. 将 Tensor 转换为 NumPy 数组
    return z_mean.asnumpy()


def avg_vector(model, data):
    latent = give_me_latent(model, data)
    arithmatic = np.average(latent, axis=0)
    return arithmatic


def reconstruct(model, data, use_data=False):
    """
    使用给定的模型和数据进行重构。

    参数:
    - model: 已训练的 VAE 模型（继承自 mindspore.nn.Cell）
    - data: 输入数据，类型可以是 NumPy 数组或其他可转换为 Tensor 的格式
    - use_data: 是否直接使用数据作为潜在向量

    返回:
    - reconstructed: NumPy 数组形式的重构数据
    """
    # 1. 设置模型为评估模式
    model.set_train(False)

    # 2. 根据 use_data 选择 latent
    if use_data:
        # 直接使用输入数据作为 latent 向量
        latent_tensor = Tensor(data, dtype=ms.float32)
    else:
        # 使用 give_me_latent 函数获取 latent 向量
        latent_np = give_me_latent(model, data)  # 假设 give_me_latent 已定义并迁移到 MindSpore
        latent_tensor = Tensor(latent_np, dtype=ms.float32)

    # 3. 进行解码
    reconstructed_tensor = model.decode(latent_tensor)

    # 4. 将重构结果转换为 NumPy 数组并返回
    return reconstructed_tensor.asnumpy()


def sample(model, n_sample):
    """
    使用给定的模型生成样本。

    参数:
    - model: 已训练的 VAE 模型（继承自 mindspore.nn.Cell）
    - n_sample: 需要生成的样本数量

    返回:
    - gen_cells: 生成的数据，NumPy 数组形式
    """
    # 1. 设置模型为评估模式
    model.set_train(False)

    # 2. 生成噪声张量
    noise_np = np.random.randn(n_sample, z_dim).astype(np.float32)
    noise_tensor = Tensor(noise_np)

    # 3. 进行解码
    gen_cells_tensor = model.decode(noise_tensor)

    # 4. 将生成的 Tensor 转换为 NumPy 数组并返回
    return gen_cells_tensor.asnumpy()


# def train(model, optimizer, data, n_epochs, batch_size=32, full_training=True, initial_run=True):
#     model.set_train()
#     data_size = data.shape[0]
#     num_batches = data_size // batch_size
#     for epoch in range(n_epochs):
#         permutation = np.random.permutation(data_size)
#         data = data[permutation, :]
#         train_loss = 0
#         for i in range(0, data_size, batch_size):
#             batch_data = data[i:i + batch_size]
#             batch_data = Tensor(batch_data, dtype=ms.float32)
#             optimizer.zero_grad()
#             x_hat, mu, log_var = model(batch_data)
#             recon_loss = 0.5 * ops.mse_loss(x_hat, batch_data, reduction='sum')
#             kl_loss = 0.5 * ops.sum(ops.exp(log_var) + mu ** 2 - 1. - log_var)
#             vae_loss = recon_loss + 0.00005 * kl_loss
#             vae_loss.backward()
#             optimizer.step()
#             train_loss += vae_loss.item()
#         avg_loss = train_loss / data_size
#         print(f"Epoch [{epoch + 1}/{n_epochs}], Loss: {avg_loss:.4f}")
#     # Save model
#     save_checkpoint(model.trainable_params(), model_to_use)
#     print(f"模型已保存到 {model_to_use}")
def train(model, optimizer, data, n_epochs, batch_size=32, full_training=True, initial_run=True):
    model.set_train()  # 设置模型为训练模式
    data_size = data.shape[0]
    
    for epoch in range(n_epochs):
        permutation = np.random.permutation(data_size)  # 打乱数据
        data = data[permutation, :]
        train_loss = 0
        
        for i in range(0, data_size, batch_size):
            # 获取批次数据并转换为 MindSpore tensor
            batch_data_np = data[i:i + batch_size]
            batch_data = Tensor(batch_data_np, mstype.float32)
            
            # 前向传播和计算损失
            def forward_fn(batch_data):
                x_hat, mu, log_var = model(batch_data)
                recon_loss = 0.5 * ops.reduce_sum(ops.mse_loss(x_hat, batch_data))
                kl_loss = 0.5 * ops.reduce_sum(ops.exp(log_var) + ops.square(mu) - 1 - log_var)
                vae_loss = recon_loss + 0.00005 * kl_loss
                return vae_loss

            # 计算梯度并更新模型参数
            grads = ops.GradOperation(get_by_list=True)(forward_fn, model.trainable_params())(batch_data)
            optimizer(grads)  # 更新参数
            
            # 计算损失值
            vae_loss = forward_fn(batch_data)
            train_loss += vae_loss.asnumpy()
        
        avg_loss = train_loss / data_size
        print(f"Epoch [{epoch + 1}/{n_epochs}], Loss: {avg_loss:.4f}")
    
    # 保存模型参数到检查点文件
    save_checkpoint(model.trainable_params(), model_to_use)
    print(f"模型已保存到 {model_to_use}")


# 定义损失函数和相关操作
# reduce_sum = ops.ReduceSum()


# # 定义损失计算函数
# def compute_loss(model, batch_data):
#     x_hat, mu, log_var = model(batch_data)
#     recon_loss = 0.5 * ops.mse_loss(x_hat, batch_data, reduction='sum')
#     kl_loss = 0.5 * (ops.Exp(log_var) + mu ** 2 - 1 - log_var).sum()
#     vae_loss = recon_loss + 0.00005 * kl_loss
#     return vae_loss


# # 定义梯度函数
# grad_fn = ops.value_and_grad(compute_loss, None, optimizer.parameters, has_aux=False)


# # 定义训练函数
# def train(model, optimizer, data, n_epochs, batch_size=32, full_training=True, initial_run=True):
#     """
#     在 MindSpore 平台上训练 VAE 模型。

#     参数:
#     - model: 已定义并初始化的 VAE 模型（继承自 mindspore.nn.Cell）
#     - optimizer: MindSpore 优化器实例
#     - data: 输入数据，类型为 NumPy 数组
#     - n_epochs: 训练的总轮数
#     - batch_size: 每个批次的大小
#     - full_training: 是否进行完整训练（暂未使用）
#     - initial_run: 是否为初始运行（暂未使用）

#     返回:
#     - 无
#     """
#     # 将模型设置为训练模式
#     model.set_train()

#     data_size = data.shape[0]

#     for epoch in range(n_epochs):
#         # 打乱数据
#         permutation = np.random.permutation(data_size)
#         shuffled_data = data[permutation, :]

#         train_loss = 0.0

#         # 迭代每个批次
#         for i in range(0, data_size, batch_size):
#             batch_data_np = shuffled_data[i:i + batch_size]
#             batch_data = Tensor(batch_data_np, dtype=mstype.float32)

#             # 计算损失和梯度
#             loss, grads = grad_fn(model, batch_data)

#             # 应用梯度更新模型参数
#             optimizer(grads)

#             # 累积损失
#             train_loss += loss.asnumpy()

#         # 计算平均损失
#         avg_loss = train_loss / data_size
#         print(f"Epoch [{epoch + 1}/{n_epochs}], Loss: {avg_loss:.4f}")

#     # 保存模型检查点
#     save_checkpoint(model, model_to_use)
#     print(f"模型已保存到 {model_to_use}")


def vector_batch_removal(model):
    # projecting data to latent space
    latent_all = give_me_latent(model, data.X)
    latent_ann = sc.AnnData(latent_all)
    latent_ann.obs["cell_type"] = data.obs["cell_type"].tolist()
    latent_ann.obs["batch"] = data.obs["batch"].tolist()
    latent_ann.obs["sample"] = data.obs["sample"].tolist()
    unique_cell_types = np.unique(latent_ann.obs["cell_type"])
    shared_anns = []
    not_shared_ann = []
    for cell_type in unique_cell_types:
        temp_cell = latent_ann[latent_ann.obs["cell_type"] == cell_type]
        if len(np.unique(temp_cell.obs["batch"])) < 2:
            cell_type_ann = latent_ann[latent_ann.obs["cell_type"] == cell_type]
            not_shared_ann.append(cell_type_ann)
            continue
        print(cell_type)
        temp_cell = latent_ann[latent_ann.obs["cell_type"] == cell_type]
        batch_list = {}
        max_batch = 0
        max_batch_ind = ""
        batchs = np.unique(temp_cell.obs["batch"])
        for i in batchs:
            temp = temp_cell[temp_cell.obs["batch"] == i]
            if max_batch < len(temp):
                max_batch = len(temp)
                max_batch_ind = i
            batch_list[i] = temp
        max_batch_ann = batch_list[max_batch_ind]
        for study in batch_list:
            delta = np.average(max_batch_ann.X, axis=0) - np.average(batch_list[study].X, axis=0)
            batch_list[study] = batch_list[study].copy()  # 创建副本
            batch_list[study].X = delta + batch_list[study].X
        corrected = anndata.concat(list(batch_list.values()))
        shared_anns.append(corrected)
    if shared_anns:
        all_shared_ann = anndata.concat(shared_anns)
    else:
        all_shared_ann = sc.AnnData()
    if not_shared_ann:
        all_not_shared_ann = anndata.concat(not_shared_ann)
    else:
        all_not_shared_ann = sc.AnnData()
    all_corrected_data = anndata.concat([all_shared_ann, all_not_shared_ann])
    # reconstructing data to gene expression space
    corrected_data = reconstruct(model, all_corrected_data.X, use_data=True)
    corrected = sc.AnnData(corrected_data)
    corrected.obs["cell_type"] = all_corrected_data.obs["cell_type"].tolist()
    corrected.obs["study"] = all_corrected_data.obs["sample"].tolist()
    corrected.var_names = data.var_names.tolist()
    # shared cell_types
    if all_shared_ann.n_obs > 0:
        corrected_shared_data = reconstruct(model, all_shared_ann.X, use_data=True)
        corrected_shared = sc.AnnData(corrected_shared_data)
        corrected_shared.obs["cell_type"] = all_shared_ann.obs["cell_type"].tolist()
        corrected_shared.obs["study"] = all_shared_ann.obs["sample"].tolist()
        corrected_shared.var_names = data.var_names.tolist()
    else:
        corrected_shared = sc.AnnData()
    return corrected, corrected_shared


# def restore(model):
#     model.load_state_dict(torch.load(model_to_use))
#     model.eval()
#     print("Model restored from %s" % model_to_use)
def restore(model):
    """
    从全局变量 `model_to_use` 指定的检查点文件中恢复 MindSpore 模型，并将其设置为评估模式。

    参数:
    - model: 已定义的 MindSpore 模型实例（继承自 mindspore.nn.Cell）

    返回:
    - 无
    """
    # 加载检查点文件中的参数字典
    param_dict = load_checkpoint(model_to_use)

    # 将参数字典加载到模型中
    load_param_into_net(model, param_dict)

    # 将模型设置为评估模式
    model.set_train(False)

    # 打印确认信息
    print(f"模型已从 {model_to_use} 恢复")


if __name__ == "__main__":
    # 原始数据的ASW指标
    data.obs["study"] = data.obs["sample"]

    # sc.pl.umap(data, color=["celltype"], save="pancreas_cell_before.pdf", show=False)
    # sc.pl.umap(data, color=["study"], save="study_pancreas_before.pdf", show=False)
    # sc.tl.pca(data, svd_solver='arpack')
    # X_pca = data.obsm["X_pca"] * -1
    # labels = data.obs["batch"].tolist()
    # print(f" average silhouette_score for original data  :{sk.metrics.silhouette_score(X_pca, labels)}")

    data.obs["cell_type"] = data.obs["celltype"]
    train(model, optimizer, train_data, n_epochs=3)
    # restore(model)
    all_data, shared = vector_batch_removal(model)
    top_cell_types = all_data.obs["cell_type"].value_counts().index.tolist()[:7]
    if "not applicable" in top_cell_types:
        top_cell_types.remove("not applicable")
    all_data.obs["celltype"] = "others"
    for cell_type in top_cell_types:
        all_data.obs.loc[all_data.obs["cell_type"] == cell_type, "celltype"] = cell_type
    all_data.write("/home/ma-user/work/scgen_tensorflow/batch_removel_data/1.h5ad")
    print(
        "scGen batch corrected pancreas has been saved in /home/ma-user/work/scgen_tensorflow/batch_removel_data/1.h5ad")

    # 处理后的ASW指标
    # sc.pp.neighbors(all_data)
    # sc.tl.umap(all_data)
    # sc.pl.umap(all_data, title="", palette=matplotlib.rcParams["axes.prop_cycle"], color=["celltype"],
    #            save="_pancreas_cell_batched.pdf", frameon=False, show=False, legend_fontsize=18)
    # sc.pl.umap(all_data, title="", palette=matplotlib.rcParams["axes.prop_cycle"], color=["study"],
    #            save="_study_pancreas_batched.pdf", frameon=False, show=False, legend_fontsize=18)
    # os.rename(src=os.path.join(path_to_save, "umap_pancreas_cell_batched.pdf"),
    #           dst=os.path.join(path_to_save, "Fig6b_umap_scgen_celltype.pdf"))
    # os.rename(src=os.path.join(path_to_save, "umap_study_pancreas_batched.pdf"),
    #           dst=os.path.join(path_to_save, "Fig6b_umap_scgen_batch.pdf"))
    # sc.tl.pca(all_data, svd_solver='arpack')
    # X_pca = all_data.obsm["X_pca"] * -1
    # labels = all_data.obs["study"].tolist()
    # print(f" average silhouette_score for scGen  :{sk.metrics.silhouette_score(X_pca, labels)}")
