import json
import threading
import os
import time

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# logger.info("00001")
# export ASCEND_SLOG_PRINT_TO_STDOUT = 1

# logger.info("00002")


import os
import numpy as np
import random
import mindspore
import mindspore.nn as nn
import mindspore.dataset as ds
from mindnlp.modules import CRF

from model_service.model_service import SingleNodeService

from mindspore import load_checkpoint, load_param_into_net
from mindspore import context

mindspore.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")

# mindspore.set_context(op_timeout=1600)


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    mindspore.set_seed(seed)
    mindspore.dataset.config.set_seed(seed)


seed = 42
seed_everything(seed)
Max_Len = 113
Entity = ['PER', 'LOC', 'ORG', 'MISC', 'DATE']
labels_text_mp = {k:v for k, v in enumerate(Entity)}
LABEL_MAP = {'O': 0}
for i, e in enumerate(Entity):
    LABEL_MAP[f'B-{e}'] = 2 * (i+1) - 1
    LABEL_MAP[f'I-{e}'] = 2 * (i+1)


# 返回词典映射表、词数字典
def get_dict(sentences):
    max_number = 1
    char_number_dict = {}

    id_indexs = {}
    id_indexs['paddding'] = 0
    id_indexs['unknow'] = 1

    for sent in sentences:
        for c in sent:
            if c not in char_number_dict:
                char_number_dict[c] = 0
            char_number_dict[c] += 1

    for c, n in char_number_dict.items():
        if n >= max_number:
            id_indexs[c] = len(id_indexs)

    return char_number_dict, id_indexs


def get_entity(decode):
    starting = False
    p_ans = []
    for i, label in enumerate(decode):
        if label > 0:
            if label % 2 == 1:
                starting = True
                p_ans.append(([i], labels_text_mp[label // 2]))
            elif starting:
                p_ans[-1][0].append(i)
        else:
            starting = False
    return p_ans


# 处理数据
class Feature(object):
    def __init__(self, sent, label, id_indexs):
        self.id_indexs = id_indexs
        self.or_text = sent  # 文本原句
        self.seq_length = len(sent) if len(sent) < Max_Len else Max_Len
        self.labels = [LABEL_MAP[c] for c in label][:Max_Len] + [0] * (Max_Len - len(label))  # 标签
        self.token_ids = self.tokenizer(sent)[:Max_Len] + [0] * (Max_Len - len(sent))  # 文本token
        self.entity = get_entity(self.labels)

    def tokenizer(self, sent):
        token_ids = []
        for c in sent:
            if c in self.id_indexs.keys():
                token_ids.append(self.id_indexs[c])
            else:
                token_ids.append(self.id_indexs['unknow'])
        return token_ids


class GetDatasetGenerator:
    def __init__(self, data, id_indexs):
        self.features = [Feature(data[0][i], data[1][i], id_indexs) for i in range(len(data[0]))]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        feature = self.features[index]
        token_ids = feature.token_ids
        labels = feature.labels

        return (token_ids, feature.seq_length, labels)


def debug_dataset(dataset):
    dataset = dataset.batch(batch_size=16)
    for data in dataset.create_dict_iterator():
        print(data["data"].shape, data["label"].shape)
        break


def get_metric(P_ans, valid):
    predict_score = 0  # 预测正确个数
    predict_number = 0  # 预测结果个数
    totol_number = 0  # 标签个数
    for i in range(len(P_ans)):
        predict_number += len(P_ans[i])
        totol_number += len(valid.features[i].entity)
        pred_true = [x for x in valid.features[i].entity if x in P_ans[i]]
        predict_score += len(pred_true)
    P = predict_score / predict_number if predict_number > 0 else 0.
    R = predict_score / totol_number if totol_number > 0 else 0.
    f1 = (2 * P * R) / (P + R) if (P + R) > 0 else 0.
    print(f'f1 = {f1}， P(准确率) = {P}, R(召回率) = {R}')


from mindspore import ms_function
from io import BytesIO


class LSTM_CRF(nn.Cell):
    def __init__(self, embedding_num, embedding_dim, num_labels):
        super().__init__()
        self.num_labels = num_labels
        self.embedding_num = embedding_num
        self.embedding_dim = embedding_dim
        self.model_name = 'LSTM_CRF'
        self.em = nn.Embedding(vocab_size=self.embedding_num, embedding_size=self.embedding_dim, padding_idx=0)
        self.bilstm = nn.LSTM(embedding_dim, embedding_dim // 2, batch_first=True, bidirectional=True)
        self.crf_hidden_fc = nn.Dense(embedding_dim, self.num_labels)
        self.crf = CRF(self.num_labels, batch_first=True, reduction='mean')

    @ms_function
    def construct(self, ids, seq_length=None, labels=None):
        # logger.info("000")
        seq = self.em(ids)
        # logger.info("001")
        lstm_feat, _ = self.bilstm(seq)
        # logger.info("002")
        emissions = self.crf_hidden_fc(lstm_feat)
        loss_crf = self.crf(emissions, tags=labels, seq_length=seq_length)
        # logger.info("003")

        return loss_crf


# 读取文本，返回词典，索引表，句子，标签
def read_data(path):
    sentences = []
    labels = []
    with open(path, 'r', encoding='utf-8') as f:
        sent = []
        label = []
        for line in f:
            parts = line.split()
            if len(parts) == 0:
                if len(sent) != 0:
                    sentences.append(sent)
                    labels.append(label)
                sent = []
                label = []
            else:
                sent.append(parts[0])
                label.append(parts[-1])

    return (sentences, labels)

def read_data1(data):
    sentences = []
    labels = []
    sent = []
    label = []
    for line in data:
        parts = line.split()
        if len(parts) == 0:
            if len(sent) != 0:
                sentences.append(sent)
                labels.append(label)
            sent = []
            label = []
        else:
            sent.append(parts[0])
            label.append(parts[-1])

    return (sentences, labels)


class class_service(SingleNodeService):
    def __init__(self, model_name, model_path):
        self.model_name = model_name
        self.model_path = model_path
        logger.info("self.model_name:%s self.model_path: %s", self.model_name,
                    self.model_path)
        self.batch_size = 1
        self.image_label_name = []
        logger.info("01")
        seed = 42
        seed_everything(seed)
        logger.info("02")
        self.model = LSTM_CRF(embedding_num=1535, embedding_dim=256, num_labels=11)
        logger.info("12")
        # 非阻塞方式加载模型，防止阻塞超时
        self.load_model()
        self.bio = BytesIO()

    def load_model(self):

        logger.info("load network ... \n")
        logger.info("11")
        self.model.set_train(False)
        # ckpt_file = './FCN8s.ckpt'
        logger.info(self.model_path)
        logger.info("12")
        ckpt_file = self.model_path + '/lstm_crf_20_63.ckpt'
        logger.info(ckpt_file)
        # print('ckpt_file:', ckpt_file)
        logger.info("13")

        param_dict = load_checkpoint(ckpt_file)
        logger.info("14")
        load_param_into_net(self.model, param_dict)
        logger.info("load network successfully ! \n")
        # print("load network successfully !")

    # 每传输一个文件过来，就会调用一次_preprocess->_inference->_postprocess
    def _preprocess(self, input_data):

        global dev

        logger.info("Get io.BytesIO!")
        logger.info("input_data:%s", input_data)

        for k, v in input_data.items():

            logger.info("k:%s", k)
            logger.info("v:%s", v)

            for file_name, file_content in v.items():
                logger.info("file_name:%s", file_name)
                logger.info("file_content:%s", file_content)

                file_content.seek(0)
                # read_data_file_content = file_content.read().decode('utf-8')
                file_content1 = file_content.getvalue()

                # logger.info("file_content1:%s", file_content1)

                file_content12 = file_content1.decode('utf-8')
                # logger.info("file_content12:%s", file_content12)
                file_content112 = file_content12.split('\n')
                # logger.info("file_content112:%s", file_content112)
                #
                # file_content11 = file_content1.read()
                # logger.info("file_content11:%s", file_content11)
                #
                # file_content111 = file_content11.decode('utf-8')
                # logger.info("file_content111:%s", file_content111)

                # content = bytesio.getvalue()

                # print(content.decode('utf-8'))

                # logger.info("read_data_file_content:%s", read_data_file_content)
                # read_data1(data)
                # read_data_file_content["train"] = json.loads(read_data_file_content["train"])
                # read_data_file_content["dev"] = json.loads(read_data_file_content["dev"])

                # logger.info("type read_data_file_content:%s", read_data_file_content)
                logger.info("00")

                dev = read_data1(file_content112)

                # logger.info("dev:%s", dev)
                # logger.info("dev:%s", type(dev))
                # logger.info("dev[0]:%s", dev[0])
                # logger.info("dev[1]:%s", dev[1])

                # read_data_file_content = json.loads(read_data_file_content)

                # train1 = read_data_file_content["train"]
                # dev1 = read_data_file_content["dev"]

                # train1 = read_data_file_content[0]
                # dev1 = read_data_file_content[1]

                # logger.info("train1:%s", train1)
                # logger.info("dev1:%s", dev1)

                # train1 = read_data_file_content.get("train")
                # dev1 = read_data_file_content.get("dev")
                logger.info("01")

                # logger.info("train1:%s", train1)
                # logger.info("dev1:%s", dev1)

                # for key in read_data_file_content:
                #     logger.info("key:%s", key)
                #     logger.info("value:%s", str(read_data_file_content[key]))
                # print(key + ":" + str(scores_dict[key]))

                # train = read_data1(train1)
                # dev = read_data1(dev1)

                # logger.info("train:%s", train)
                # logger.info("dev:%s", dev)
                logger.info("02")

        logger.info("20")
        # char_number_dict, id_indexs = get_dict(train[0])

        logger.info("id_indexs.json load begin!")
        f = open(self.model_path + '/id_indexs.json', 'r')
        content = f.read()
        id_indexs = json.loads(content)
        f.close()
        logger.info("id_indexs.json load over!")
        # logger.info("id_indexs:%s", id_indexs)

        dev_dataset_generator = GetDatasetGenerator(dev, id_indexs)
        logger.info("21")

        dataset_dev = ds.GeneratorDataset(dev_dataset_generator, ["data", "length", "label"], shuffle=False)
        dataset_dev = dataset_dev.batch(batch_size=self.batch_size)
        size = dataset_dev.get_dataset_size()
        steps = size
        logger.info("steps:%s", steps)

        return dataset_dev

    def _inference(self, dataset_dev):
        logger.info("_inference")

        decodes = []
        logger.info("30")
        self.model.set_train(False)
        i = 0
        for batch, (token_ids, seq_length, labels) in enumerate(dataset_dev.create_tuple_iterator()):
            start = time.time()
            # logger.info("token_ids:%s", token_ids)
            logger.info("31")
            score, history = self.model(token_ids, seq_length=seq_length)
            # logger.info("score:%s", score)
            logger.info("311")
            best_tags = self.model.crf.post_decode(score, history, seq_length)
            logger.info("32")
            decode = [[y.asnumpy().item() for y in x] for x in best_tags]
            decodes.extend(list(decode))
            end = time.time()
            logger.info("33")
            i = i + 1
            logger.info("step:%s", i)
            logger.info("step time :%s", end - start)

        v_pred = [get_entity(x) for x in decodes]
        logger.info("34")
        logger.info("v_pred:%s", v_pred)

        return v_pred

    def _postprocess(self, v_pred):
        result = {}
        result["result"] = v_pred
        logger.info("result:%s", result)

        return result



