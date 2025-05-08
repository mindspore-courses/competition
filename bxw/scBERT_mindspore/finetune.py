import argparse
import numpy as np
from dataset2 import load_data
from performer_mindspore import PerformerLM
from mindspore.nn import Adam, CrossEntropyLoss
from tqdm import tqdm
from mindspore import ops, save_checkpoint, Tensor
import math
from functools import reduce
import mindspore as ms
from mindspore import value_and_grad, ParallelMode, nn
from mindspore.communication import init
from mindspore import Profiler
import pickle as pkl
from sklearn.metrics import accuracy_score
import os

# 微调中新的输出层
class Identity(nn.Cell):
    def __init__(self, dropout = 0.1, h_dim = 100, out_dim = 10):
        super(Identity, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, (1,200), pad_mode='valid', padding=0, has_bias=False)
        self.act = nn.ReLU()
        self.fc1 = nn.Dense(in_channels=SEQ_LEN, out_channels=512, has_bias=True)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Dense(in_channels=512, out_channels=h_dim, has_bias=True)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Dense(in_channels=h_dim, out_channels=out_dim, has_bias=True)

    def construct(self, x):
        x = x[:,None,:,:]
        # [batch, 1, seq_len, 200]
        x = self.conv1(x)
        # [batch, 1, seq_len, 1]
        x = self.act(x)
        x = x.view(x.shape[0],-1)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

model = None
loss_fn = None

def cum_loss_and_logits(data, label):
    global model, loss_fn, SEQ_LEN
    logits = model(data)
    # label = ops.repeat_elements(label, rep=7, axis=-1)
    # label = ops.reshape(label, (-1, SEQ_LEN, 7))
    # label = ops.Cast()(label, dtype=ms.dtype.float32)
    loss = loss_fn(logits, label)
    return loss, logits

def build_model(args):
    global CLASS, SEQ_LEN, POS_EMBED_USING, model
    #load the label stored 
    with open('label_dict', 'rb') as fp:
        label_dict = pkl.load(fp)
    with open('label', 'rb') as fp:
        label = pkl.load(fp)
    model = PerformerLM(
        num_tokens = CLASS,
        dim = 200,
        depth = 6,
        max_seq_len = SEQ_LEN,
        heads = 10,
        # local_attn_heads = 0,
        # g2v_position_emb = POS_EMBED_USING  
    )
    args = parse()
    # 加载预训练权重
    ckpt_file_name = args.model_path
    param_dict = ms.load_checkpoint(ckpt_file_name)
    # 将权重加载到模型中
    ms.load_param_into_net(model, param_dict)
    # 设置参数是否参与梯度计算
    for param in model.trainable_params():
        param.requires_grad = False
    for param in model.norm.trainable_params():
        param.requires_grad = True
    for param in model.performer.layers[-2].trainable_params():
        param.requires_grad = True
    # 覆盖输出层
    model.to_out = Identity(dropout=0.1, h_dim=128, out_dim=label_dict.shape[0])
    print("build model success.")
    count = sum([ item.size for item in model.get_parameters()])
    names = [item.name for item in model.trainable_params()]
    print("param count is {}, names: {}, count: {}".format(count, str(names), len(names)))
    
    if args.enable_pipeline:
        model.init_pipeline(0)
        model.performer.layers[0].init_pipeline(1)
        model.performer.layers[0].attention.init_pipeline(1)
    return

def build_optimizer_and_scheduler(model):
    global LEARNING_RATE, PAD_TOKEN_ID, loss_fn, optimizer
    # optimizer
    optimizer = Adam(params=model.trainable_params(), learning_rate=LEARNING_RATE)
    # loss
    loss_fn = CrossEntropyLoss(weight=None)
    print("build optimizer success.")
    return optimizer

def train_one_epoch(train_dataloader, grad_fn, optimizer):
    global model
    running_loss = 0.0
    cum_acc = 0.0
    model.set_train(True)
    for _, (data, label) in enumerate(tqdm(train_dataloader.create_tuple_iterator())):
        # forward 推理
        (loss, logits), grads = grad_fn(data, label)
        optimizer(grads)
        # 累加损失
        running_loss += loss.item() 
        # 计算精度
        final = ops.softmax(logits)
        final = final.argmax(axis=-1)
        # 预测数
        pred_num = Tensor([final.shape[-1]], ms.int32) 
        # 计算正确数
        correct_num = ops.Equal()(final, label).sum(axis=-1)
        # 计算累计准确率
        cum_acc += correct_num / pred_num.mean()
        del data, label, final
        # profiler.analyse()
        
    return running_loss, cum_acc

# 从 Tensor 对象中提取整数值
def get_value_from_tensor(tensor_list):
    return [tensor.asnumpy()[0] for tensor in tensor_list]

def eval_one_epoch(val_dataloader):
    global loss_fn, model, SEQ_LEN
    model.set_train(False)
    predictions = []
    truths = []
    running_loss = 0.0
    print("========== 开始验证")
    for _, (data,label) in enumerate(tqdm(val_dataloader.create_tuple_iterator())): 
        logits = model(data)
        loss = loss_fn(logits, label)
        running_loss += loss.item()
        softmax = nn.Softmax(axis=-1)
        final_prob = softmax(logits)
        final = final_prob.argmax(axis=-1)
        predictions.append(final)
        truths.append(label)
        del data, logits, final
    val_loss = running_loss / len(val_dataloader)
    # 获取 truths 和 predictions 的实际值
    truths_values = get_value_from_tensor(truths)
    predictions_values = get_value_from_tensor(predictions)
    # 计算正确率
    correct_count = sum(t == p for t, p in zip(truths_values, predictions_values))
    total_count = len(truths_values)
    val_acc = correct_count / total_count if total_count > 0 else 0
    # 计算正确数
    del predictions, truths
    return val_loss, val_acc

def train(optimizer, train_dataloader, val_dataloader):
    global EPOCHS,VALIDATE_EVERY, MODEL_NAME, loss_fn
    
    train_num_step =  len(train_dataloader)
    grad_fn = value_and_grad(cum_loss_and_logits, grad_position=None, weights=model.trainable_params(), has_aux=True)
    for epoch in range(EPOCHS):
        running_loss, cum_acc = train_one_epoch(train_dataloader, grad_fn, optimizer)
        # log epoch的信息
        epoch_loss = running_loss / train_num_step
        epoch_acc = 100 * cum_acc / train_num_step

        # 确保将Tensor转换为Python数值
        epoch_loss_value = epoch_loss.asnumpy().item() if isinstance(epoch_loss, ms.Tensor) else epoch_loss
        epoch_acc_value = epoch_acc.asnumpy().item() if isinstance(epoch_acc, ms.Tensor) else epoch_acc

        log_string = f'    ==  Epoch: {epoch} | Training Loss: {epoch_loss_value:.6f} | Accuracy: {epoch_acc_value:6.4f}%  =='
        print(log_string)
        with open('finetune_result.txt', 'a') as f:  
            f.write(log_string  + '\n')

        # 进行一次验证
        if epoch % VALIDATE_EVERY == 0:
            val_loss, val_acc = eval_one_epoch(val_dataloader)
            log_string = f'    ==  Epoch: {epoch} | Validation Loss: {val_loss} | Accuracy: {val_acc.item()}%  =='
            print(log_string)
            with open('finetune_result.txt', 'a') as f:  
                f.write(log_string  + '\n')

        ckpt_dir = "./" + FINETUNE_SAVE_PATH
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_file = f"finetune-{epoch}.ckpt"
        ckpt_path = os.path.join(ckpt_dir, ckpt_file)
        save_checkpoint(model, ckpt_path)

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--enable_pipeline", type=bool, default=False, help='Local process rank.')
    parser.add_argument("--device_id", type=int, default=-1, help='Local process rank.')
    parser.add_argument("--bin_num", type=int, default=5, help='Number of bins.')
    parser.add_argument("--gene_num", type=int, default=16906, help='Number of genes.')
    parser.add_argument("--epoch", type=int, default=100, help='Number of epochs.')
    parser.add_argument("--seed", type=int, default=2021, help='Random seed.')
    parser.add_argument("--batch_size", type=int, default=1, help='Number of batch size.')
    parser.add_argument("--learning_rate", type=float, default=1e-4, help='Learning rate.')
    parser.add_argument("--grad_acc", type=int, default=60, help='Number of gradient accumulation.')
    parser.add_argument("--valid_every", type=int, default=1, help='Number of training epochs between twice validation.')
    parser.add_argument("--pos_embed", type=bool, default=True, help='Using Gene2vec encoding or not.')
    parser.add_argument("--data_path", type=str, default='./data/Zheng68k_prepeocessed.h5ad', help='Path of data for finetune.')
    parser.add_argument("--model_path", type=str, default='./pretrain_ckpts/pretrain-0.ckpt', help='Path of pretrained model.')
    parser.add_argument("--ckpt_dir", type=str, default='./finetune_ckpts/', help='Directory of checkpoint to save.')
    parser.add_argument("--model_name", type=str, default='finetune', help='Finetuned model name.')
    args = parser.parse_args()
    return args
     
if __name__ == "__main__":
    # 1. 解析命令行参数
    args = parse()
    if args.enable_pipeline:
        ms.set_context(mode=0, device_target="Ascend")
        ms.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL, pipeline_stages=2, pipeline_result_broadcast=True)
        init()
        ms.set_seed(1)
    else:
        ms.set_context(variable_memory_max_size='29GB')
        ms.set_context(mode=0, device_target="Ascend", device_id=0)
    # 2. 声明全局变量
    SEED = args.seed
    EPOCHS = args.epoch
    BATCH_SIZE = args.batch_size
    GRADIENT_ACCUMULATION = args.grad_acc
    LEARNING_RATE = args.learning_rate
    SEQ_LEN = args.gene_num + 1
    VALIDATE_EVERY = args.valid_every
    PATIENCE = 10
    UNASSIGN_THRES = 0.0
    CLASS = args.bin_num + 2
    POS_EMBED_USING = args.pos_embed
    FINETUNE_SAVE_PATH = args.ckpt_dir
    # 3. 加载数据集
    train_dataloader, val_dataloader = load_data(args.data_path, CLASS, SEED, BATCH_SIZE)
    # 4. 加载模型
    build_model(args)
    # 4. 构建优化器和损失函数
    optimizer = build_optimizer_and_scheduler(model)
    # 5. 开始训练
    train(optimizer, train_dataloader, val_dataloader)
