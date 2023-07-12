import json
import time
import logging
import os
import numpy as np
import random

import mindspore
import mindspore.nn as nn
import mindspore.dataset as ds
from mindspore import load_checkpoint, load_param_into_net
from mindspore import context
from mindspore import ms_function

from mindnlp.modules import CRF

from model_service.model_service import SingleNodeService


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

mindspore.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    mindspore.set_seed(seed)
    mindspore.dataset.config.set_seed(seed)


seed = 42
seed_everything(seed)
Max_Len = 113
# 标签：人物PER， 地点LOC，机构ORG，混杂类型MISC（miscellaneous），日期DATE
Entity = ['PER', 'LOC', 'ORG', 'MISC', 'DATE']
labels_text_mp = {k: v for k, v in enumerate(Entity)}
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

    @ms_function  # 图模式
    def construct(self, ids, seq_length=None, labels=None):
        seq = self.em(ids)
        lstm_feat, _ = self.bilstm(seq)
        emissions = self.crf_hidden_fc(lstm_feat)
        loss_crf = self.crf(emissions, tags=labels, seq_length=seq_length)

        return loss_crf


# 读取文本，返回词典，索引表，句子，标签
def read_data(data):
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
        self.model = LSTM_CRF(embedding_num=1535, embedding_dim=256, num_labels=11)
        # 非阻塞方式加载模型，防止阻塞超时
        self.load_model()

    def load_model(self):
        # 模型加载
        logger.info("load network ... \n")
        logger.info(self.model_path)
        ckpt_file = self.model_path + '/lstm_crf_20_63.ckpt'
        logger.info(ckpt_file)
        param_dict = load_checkpoint(ckpt_file)
        load_param_into_net(self.model, param_dict)
        logger.info("load network successfully ! \n")

    # 读取已经上传好的测试数据集，一个文件，一个文件的读取
    # 每传输一个文件过来，就会调用一次_preprocess->_inference->_postprocess，就会生成一个结果文件
    def _preprocess(self, input_data):
        # 读取测试数据集，modelarts自动读取文件
        global dev
        logger.info("Get io.BytesIO!")
        logger.info("input_data:%s", input_data)
        for k, v in input_data.items():
            logger.info("k:%s", k)
            logger.info("v:%s", v)
            for file_name, file_content in v.items():
                logger.info("file_name:%s", file_name)
                file_content.seek(0)
                file_content1 = file_content.getvalue().decode('utf-8').split('\n')
                dev = read_data(file_content1)

        # 读取id_indexs数据
        logger.info("id_indexs.json load begin!")
        f = open(self.model_path + '/id_indexs.json', 'r')
        content = f.read()
        id_indexs = json.loads(content)
        f.close()
        logger.info("id_indexs.json load over!")

        # 数据处理
        dev_dataset_generator = GetDatasetGenerator(dev, id_indexs)
        dataset_dev = ds.GeneratorDataset(dev_dataset_generator, ["data", "length", "label"], shuffle=False)
        dataset_dev = dataset_dev.batch(batch_size=self.batch_size)
        size = dataset_dev.get_dataset_size()
        logger.info("size:%s", size)

        return dataset_dev

    def _inference(self, dataset_dev):
        # 模型推理
        logger.info("_inference")
        decodes = []
        self.model.set_train(False)
        i = 0
        for batch, (token_ids, seq_length, labels) in enumerate(dataset_dev.create_tuple_iterator()):
            start = time.time()
            score, history = self.model(token_ids, seq_length=seq_length)
            best_tags = self.model.crf.post_decode(score, history, seq_length)
            decode = [[y.asnumpy().item() for y in x] for x in best_tags]
            decodes.extend(list(decode))
            end = time.time()
            i = i + 1
            logger.info("step:%s", i)
            logger.info("step time :%s", end - start)

        v_pred = [get_entity(x) for x in decodes]
        logger.info("v_pred:%s", v_pred)

        return v_pred

    def _postprocess(self, v_pred):
        # 讲结果包装进字典
        result = {}
        result["result"] = v_pred
        logger.info("result:%s", result)

        return result



