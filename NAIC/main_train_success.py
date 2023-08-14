# -*- coding: utf-8 -*-
"""
@Time:2023/8/9 10:04
@Author : zwx1172307\Joegame\周剑
"""

import time, os
import numpy as np
import scipy.io as scio

import mindspore as ms
from mindspore import nn, ops
import mindspore.dataset as ds
from mindspore import save_checkpoint

ms.set_context(device_target='CPU', mode=ms.PYNATIVE_MODE)

# DL Parameters
total_epoch = 150
LR = 1e-4
batch_size = 64
BS = 2
pretrain = False
ckpt_path = './ckpt/modelBS2_99_4.603.ckpt'


checkpoints_path = f"./ckpt/ckpt202308111737_{BS}"
print('checkpoints_path: ', checkpoints_path)

print('BS: ', BS)


# Read Data
if BS == 1:
    print('./dataset/data_train_1.mat')
    data = scio.loadmat('./dataset/data_train_1.mat')
    location_data_BS1 = data['loc_1']  # ndarray:(2400, 3)
    channel_data_BS1 = data['CSI_1']  # ndarray:(2400, 8, 32, 120, 2)
    data = {'1': location_data_BS1, '2': channel_data_BS1}
else:
    print('./dataset/data_train_2.mat')
    data = scio.loadmat('./dataset/data_train_2.mat')
    location_data_BS2 = data['loc_2']
    channel_data_BS2 = data['CSI_2']
    data = {'1': location_data_BS2, '2': channel_data_BS2}

dataset = ds.NumpySlicesDataset(data=data, column_names=["data", "label"], shuffle=True)

datasize = dataset.get_dataset_size()

print('datasize: ', datasize)

dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
step = dataset.get_dataset_size()

print('step: ', step)


# MIMO-OFDM Parameters
SC_num = 120  # subcarrier number
Tx_num = 32  # Tx antenna number
Rx_num = 8  # Rx antenna number
sigma2_UE = 1e-6


class NeuralNetwork(nn.Cell):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Dense(in_channels=3, out_channels=50, activation='sigmoid')
        self.bn1 = nn.BatchNorm1d(50)
        self.layer2 = nn.Dense(in_channels=50, out_channels=100, activation='sigmoid')
        self.bn2 = nn.BatchNorm1d(100)
        self.layer3 = nn.Dense(in_channels=100, out_channels=Rx_num * Tx_num * SC_num * 2, activation='sigmoid')

    def construct(self, x):
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.layer2(x)
        x = self.bn2(x)
        x = self.layer3(x)

        return x


class NeuralNetwork1(nn.Cell):
    def __init__(self):
        super(NeuralNetwork1, self).__init__()
        self.layer1 = nn.Dense(in_channels=3, out_channels=50, weight_init='xavier_uniform')
        self.bn1 = nn.BatchNorm1d(50, eps=0.001, momentum=0.99)
        self.layer2 = nn.Dense(in_channels=50, out_channels=100, weight_init='xavier_uniform')
        self.bn2 = nn.BatchNorm1d(100, eps=0.001, momentum=0.99)
        self.layer3 = nn.Dense(in_channels=100, out_channels=Rx_num * Tx_num * SC_num * 2, weight_init='xavier_uniform')

    def construct(self, x):
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.layer2(x)
        x = self.bn2(x)
        x = self.layer3(x)

        return x


net = NeuralNetwork1()

print('pretrain: ', pretrain)

if pretrain:
    print('ckpt_path: ', ckpt_path)
    parameter = ms.load_checkpoint(ckpt_path)
    ms.load_param_into_net(net, parameter)

loss_fn = nn.MSELoss()
opt = nn.Adam(params=net.trainable_params(), learning_rate=LR)


def forward_fn(data):
    logits = net(data['data'])
    x = ops.reshape(logits, (batch_size, Rx_num, Tx_num, SC_num, 2))
    loss = loss_fn(x, data['label'])
    return loss


# 梯度方法
grad_fn = ops.value_and_grad(forward_fn, None, net.trainable_params())


def train_step(data_1):
    # 计算判别器损失和梯度
    loss_1, grads = grad_fn(data_1)
    opt(grads)
    return loss_1


os.makedirs(checkpoints_path, exist_ok=True)


def train():
    for epoch in range(total_epoch):
        loss_all = []
        start = time.time()
        for iter, data in enumerate(dataset.create_dict_iterator()):
            start1 = time.time()
            loss = train_step(data)
            loss_all.append(loss.asnumpy())
            end1 = time.time()
            if iter % 10 == 0:
                print(f"Epoch:[{int(epoch):>3d}/{int(total_epoch):>3d}], "
                      f"step:[{int(iter):>4d}/{int(step):>4d}], "
                      f"loss:{loss.asnumpy():.4e} , "
                      f"time:{(end1 - start1):>3f}s, ")

        end = time.time()
        print("time of epoch {} is {:.2f}s".format(epoch + 1, end - start))
        print("mean loss: ", sum(loss_all)/len(loss_all))

        # 根据epoch保存模型权重文件
        if (epoch+1) % 10 == 0:
            save_checkpoint(net, checkpoints_path + f"/modelBS{int(BS)}_{epoch}.ckpt")


train()
