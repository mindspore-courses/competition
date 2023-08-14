# -*- coding: utf-8 -*-
"""
@Time:2023/8/9 10:04
@Author : zwx1172307\Joegame\周剑
"""

import numpy as np
import scipy.io as scio

import mindspore as ms
from mindspore import nn, ops
import mindspore.dataset as ds

ms.set_context(device_target='CPU', mode=ms.PYNATIVE_MODE)

# DL Parameters
# LR = 5e-4
ckpt_path1 = './ckpt/modelBS1_79_4.80.ckpt'
ckpt_path2 = './ckpt/modelBS2_79_4.80.ckpt'


# MIMO-OFDM Parameters
SC_num = 120  # subcarrier number
Tx_num = 32  # Tx antenna number
Rx_num = 8  # Rx antenna number
sigma2_UE = 1e-6


class NeuralNetwork(nn.Cell):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
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


def RadioMap_Model1(data, net):
    CSI_est = net(data['data'])
    HH_est = ops.reshape(CSI_est, (-1, Rx_num, Tx_num, SC_num, 2))
    HH_complex_est = ops.Complex()(HH_est[:, :, :, :, 0], HH_est[:, :, :, :, 1])
    HH_complex_est = ops.transpose(HH_complex_est, (0, 3, 1, 2))
    MatDiag, MatRx, MatTx = np.linalg.svd(HH_complex_est.asnumpy(), full_matrices=True)

    PrecodingVector = MatTx[:, :, :, 0]
    PrecodingVector = np.reshape(PrecodingVector, (-1, SC_num, Tx_num, 1))
    return PrecodingVector


def EqChannelGainJoint(Channel_BS1, Channel_BS2, PrecodingVector_BS1, PrecodingVector_BS2):
    # The authentic CSI
    # HH1
    HH1 = np.reshape(Channel_BS1, (-1, Rx_num, Tx_num, SC_num, 2)) ## Rx, Tx, Subcarrier, RealImag
    HH1_complex = HH1[:,:,:,:,0] + 1j * HH1[:,:,:,:,1]  ## Rx, Tx, Subcarrier
    HH1_complex = np.transpose(HH1_complex, [0,3,1,2])

    # HH2
    HH2 = np.reshape(Channel_BS2, (-1, Rx_num, Tx_num, SC_num, 2))  # Rx, Tx, Subcarrier, RealImag
    HH2_complex = HH2[:,:,:,:,0] + 1j * HH2[:,:,:,:,1]  # Rx, Tx, Subcarrier
    HH2_complex = np.transpose(HH2_complex, [0,3,1,2])

    # Power Normalization of the precoding vector
    # PrecodingVector1
    Power = np.matmul(np.transpose(np.conj(PrecodingVector_BS1), (0, 1, 3, 2)), PrecodingVector_BS1)
    Power = np.sum(Power.reshape(-1, SC_num), axis=-1).reshape(-1, 1)
    Power = np.matmul(Power, np.ones((1, SC_num)))
    Power = Power.reshape(-1, SC_num, 1, 1)
    PrecodingVector_BS1 = np.sqrt(SC_num) * PrecodingVector_BS1 / np.sqrt(Power)

    # PrecodingVector2
    Power = np.matmul(np.transpose(np.conj(PrecodingVector_BS2), (0, 1, 3, 2)), PrecodingVector_BS2)
    Power = np.sum(Power.reshape(-1, SC_num), axis=-1).reshape(-1, 1)
    Power = np.matmul(Power, np.ones((1, SC_num)))
    Power = Power.reshape(-1, SC_num, 1, 1)
    PrecodingVector_BS2 = np.sqrt(SC_num) * PrecodingVector_BS2 / np.sqrt(Power)

    # Effective channel gain
    R = np.matmul(HH1_complex, PrecodingVector_BS1) + np.matmul(HH2_complex, PrecodingVector_BS2)
    R_conj = np.transpose(np.conj(R), (0, 1, 3, 2))
    h_sub_gain = np.matmul(R_conj, R)
    h_sub_gain = np.reshape(np.absolute(h_sub_gain), (-1, SC_num))  # channel gain of SC_num subcarriers
    return h_sub_gain


# Data Rate
def DataRate(h_sub_gain, sigma2_UE):  ### Score
    SNR = h_sub_gain / sigma2_UE
    Rate = np.log2(1 + SNR)  ## rate
    Rate_OFDM = np.mean(Rate, axis=-1)  ###  averaging over subcarriers
    Rate_OFDM_mean = np.mean(Rate_OFDM)  ### averaging over CSI samples
    return Rate_OFDM_mean


# eval
# Parameters
batch_size = 1
# sigma2_UE = 1e-6


# Load data

data = scio.loadmat('./dataset/data_test.mat')
location_data = data['loc'].astype(np.float32)  # ndarray:(1854, 8, 32, 120, 2)
channel_data_1 = data['CSI_1'].astype(np.float32)  # ndarray:(1854, 8, 32, 120, 2)
channel_data_2 = data['CSI_2'].astype(np.float32)  # ndarray:(1854, 8, 32, 120, 2)

data = {'1': location_data, '2': channel_data_1, '3': channel_data_2}
dataset = ds.NumpySlicesDataset(data=data, column_names=["data", "label1", "label2"], shuffle=True)
datasize = dataset.get_dataset_size()
print('datasize eval: ', datasize)
dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
step = dataset.get_dataset_size()
print('step eval: ', step)


# Load model
net1 = NeuralNetwork()
parameter = ms.load_checkpoint(ckpt_path1)
ms.load_param_into_net(net1, parameter)

net2 = NeuralNetwork()
parameter = ms.load_checkpoint(ckpt_path2)
ms.load_param_into_net(net2, parameter)

data_rates = []

for iter, data in enumerate(dataset.create_dict_iterator()):
    result1 = RadioMap_Model1(data, net1)  # StubTensor:(64, 120, 32, 1)
    result2 = RadioMap_Model1(data, net2)  # StubTensor:(64, 120, 32, 1)

    # Calculate the score
    d1 = data['label1'].asnumpy()
    d2 = data['label2'].asnumpy()

    SubCH_gain_codeword = EqChannelGainJoint(d1, d2, result1, result2)

    data_rate = DataRate(SubCH_gain_codeword, sigma2_UE)
    data_rates.append(data_rate)
    if iter % 100 == 0:
        print(f'{iter} step')
        print('The score is %f bps/Hz' % data_rate)

score = sum(data_rates)/len(data_rates)

print('The mean score is %f bps/Hz' % score)
