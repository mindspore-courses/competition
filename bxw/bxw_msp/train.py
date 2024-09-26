import mindspore
import mindspore.nn as nn
import scipy as sp
from tqdm.auto import tqdm
from mindspore.dataset import GeneratorDataset
from mindspore import context, ops, Parameter, Tensor
from mindspore.nn.probability.distribution import Normal, Distribution, Bernoulli
from mindspore.ops import broadcast_to, log as ms_log
from mindspore.train.callback import LossMonitor
from mindspore.common import dtype as mstype
from mindspore import value_and_grad
from typing import Any, Dict, List, Literal, Iterable, Optional, Sequence, Tuple, Union, Mapping, TypedDict
from config import *
from network import *
from loss import *
from distribution import *
from dataloader import *
# context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
# mindspore.set_context(pynative_synchronize=True, runtime_num_threads=1)

def train(config: Dict):
    data_module = ReplogleDataModule(batch_size=config["data_module_kwargs"]["batch_size"])
    config = add_data_info_to_config(config, data_module)

    # 设置随机种子
    mindspore.set_seed(config["seed"])

    # Step 1: 初始化模型、引导模块、损失模块和优化器
    net = Net(config, data_module)

    # loss_module = SAMSVAE_ELBOLossModule()
    # loss_fn = loss_module.loss_fn

    # 手动指定需要优化的参数
    params_to_pass = [
        net.model.decoder.log_concentration,
        net.model.decoder.mlp.layers[0][0].weight,
        net.model.decoder.mlp.layers[0][0].bias,
        net.model.decoder.normalized_mean_decoder[0].weight,
        net.model.decoder.normalized_mean_decoder[0].bias,
        net.guide.q_mask_logits,
        net.guide.q_E_loc,
        net.guide.q_E_log_scale,
        net.guide.z_basal_encoder.mlp.layers[0][0].weight,
        net.guide.z_basal_encoder.mlp.layers[0][0].bias,
        net.guide.z_basal_encoder.mean_encoder.weight,
        net.guide.z_basal_encoder.mean_encoder.bias,
        net.guide.z_basal_encoder.log_var_encoder.weight,
        net.guide.z_basal_encoder.log_var_encoder.bias
    ]

    # 将这些参数传给优化器
    optimizer = nn.Adam(params=params_to_pass, learning_rate=config["lightning_module_kwargs"]["lr"])
    obs_count = data_module.get_train_perturbation_obs_counts()

    def train_step(batch, n_particles, condition_values):
        grad_op = ops.GradOperation(get_by_list=True)
        params = params_to_pass
        # https://www.mindspore.cn/docs/zh-CN/r2.3.1/api_python/ops/mindspore.ops.GradOperation.html
        loss, grad = grad_op(net,params)(batch, n_particles, obs_count, condition_values) 
        optimizer(grad)
        return loss

    def train_one_epoch(train_dataloader, config):
        total_loss = 0
        num_batches = 0
        cast = ops.Cast()
        n_particles = config["lightning_module_kwargs"]["n_particles"]
        for batch in train_dataloader:
            idx, X, D, library_size = batch  # 解包列表
            library_size = Tensor(library_size, mstype.float32)
            condition_values = {'idx':idx, 'library_size':library_size}
            loss = train_step(batch, n_particles, condition_values)
            total_loss += loss.asnumpy()
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"Average training loss: {avg_loss}")
        return avg_loss

    def validate(val_dataloader, model, guide, loss_module):
        model.set_train(False)
        val_loss = 0
        num_batches = 0

        for batch in val_dataloader:
            X = batch['X']
            D = batch['D']
            D_obs_counts = batch.get('library_size', None)
            
            with mindspore.no_grad():
                loss, _ = loss_module.loss(X, D, D_obs_counts, n_particles=config["lightning_module_kwargs"]["n_particles"])
            val_loss += loss.asnumpy()
            num_batches += 1
        
        avg_val_loss = val_loss / num_batches
        print(f"Validation loss: {avg_val_loss}")
        return avg_val_loss

    train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader()

    # Step 4: 训练和验证模型
    for epoch in range(config["max_epochs"]):
        print(f"Epoch {epoch + 1}/{config['max_epochs']}")
        net.set_train()
        train_loss = train_one_epoch(train_dataloader, config)
        print("SUCCESSFUL!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # val_loss = validate(val_dataloader, model, guide, loss_module)


        # 可选：每个epoch后保存模型
        # mindspore.save_checkpoint(model, f"samsvae_model_epoch_{epoch+1}.ckpt")

    # Step 5: 保存最终模型
    # mindspore.save_checkpoint(model, "samsvae_model_final.ckpt")

    # Step 6: 加载模型（如有需要）
    # mindspore.load_checkpoint("samsvae_model_final.ckpt", net=model)

    # 可选：预测步骤可以根据config["predictor"]来实现


if __name__ == "__main__":
    train(config)