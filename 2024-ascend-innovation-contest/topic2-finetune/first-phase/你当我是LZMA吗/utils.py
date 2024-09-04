#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/07/05 

import sys
import json
from re import compile as Regex
from pathlib import Path
from typing import List, Tuple, Dict, Any

import random
import numpy as np
SEED = 114514
random.seed(SEED)
np.random.seed(SEED)

IS_WIN = sys.platform == 'win32'

BASE_PATH = Path(__file__).parent
if IS_WIN:
  DATASET_RAW_FILE = BASE_PATH / 'material' / 'train.json'
  DATASET_PROC_FILE = BASE_PATH / 'material' / 'train-data-conversation.json'
else:
  DATASET_RAW_FILE = BASE_PATH / 'train.json'
  DATASET_PROC_FILE = BASE_PATH / 'train-data-conversation.json'
DATASET_TEST_FILE = BASE_PATH / 'data_200_random.json'

DEFAULT_SYSTEM_PROMPT = 'Below is an instruction that describes a task. Write a response that appropriately completes the request.'
PROMPT_TEMPLATE = {
  "id": "[sample_id]", 
  "conversations": [
    {
      "from": "human", 
      "value": f"{DEFAULT_SYSTEM_PROMPT}\n\n### Instruction:\n[problem]\n\n### Response:"}, 
    {
      "from": "gpt", 
      "value": "[solution]",
    }
  ]
}

QA = Tuple[str, str]

PROBLEM_TEMPLATES = [{
  # 359347 cases
  'id': 0,
  'note': '简单四则运算 A op B，有效数字：+(2), -(2), *(4), /(inf)',
  'Q': Regex('计算 ([\d\. \+\-\*\/]+) 等于多少？'),
  'A': Regex('([\d\. \+\-\*\/]+) = (-?[\d\.]+)'),
}, {
  # 39207 cases
  'id': 1,
  'note': '求幂 A^n，有效数字 2*n',
  'Q': Regex('计算 (-?[\d\.]+) 的 (\d+) 次方？'),
  'A': Regex('(-?[\d\.]+)\^(\d+) = (-?[\d\.]+)'),
}, {
  # 39227 cases
  'id': 2,
  'note': '求平方根，有效数字 inf',
  'Q': Regex('计算 ([\d\.]+) 的平方根？'),
  'A': Regex('√([\d\.]+) = ([\d\.]+)'),
}, {
  # 45 cases
  'id': 3,
  'note': '整除运算',
  'Q': Regex('将分数 (\d+)/(\d+) 进行简化。'),
  'A': Regex('最简化的形式为：(\d+)/(\d+)'),
}, {
  # 19709 cases
  'id': 4,
  'note': '求均值，有效数字 inf',
  'Q': Regex('求以下数据的平均值：(\[[\d, ]+\])'),
  'A': Regex('平均值为 ([\d\.]+)'),
}, {
  # 40080 cases
  'id': 5,
  'note': '除法运算，有效数字 inf',
  'Q': Regex('解方程 (-?\d+)x \+ (-?\d+) = 0'),
  'A': Regex('方程的解为：(-?[\d\.]+)'),
}, {
  # 20028 cases
  'id': 6,
  'note': '乘法+幂运算，有效数字 inf',
  'Q': Regex('当 x = (-?[\d\.]+) 时，求函数 y = (-?\d+)x\^(\d+) 的值'),
  'A': Regex('函数的值为：([\d\.E\+\-]+)'),
}, {
  # 8665 cases
  'id': 7,
  'note': '乘法运算',
  'Q': Regex('一个长方形的长为 (\d+) 厘米，宽为 (\d+) 厘米，请计算其面积。'),
  'A': Regex('面积为 (\d+) 平方厘米'),
}, {
  # 100 cases
  'id': 8,
  'note': '乘法运算',
  'Q': Regex('某物体的密度为 (\d+) 克/立方厘米，体积为 (\d+) 立方厘米，请计算该物体的质量。'),
  'A': Regex('(\d+) 克'),
}, {
  # 6275 cases
  'id': 9,
  'note': '减法-乘法运算；线性回归，有效数字 2',
  'Q': Regex('商品原价为 (\d+) 元，打折后的价格为 (\d+) 元，请计算打折的折扣比例。'),
  'A': Regex('(-?[\d\.]+)'),
}, {
  # 6266 cases
  'id': 10,
  'note': '加法-乘法运算；线性回归，有效数字 2',
  'Q': Regex('去年销售额为 (\d+) 万元，今年销售额增加了 (\d+)%，请计算今年的销售额。'),
  'A': Regex('(-?[\d\.]+)'),
}]


# 仅返回中文数据 (已去重)
def load_dataset_raw() -> List[QA]:
  with open(DATASET_RAW_FILE, 'r', encoding='utf-8') as fh:
    cases = fh.read().strip().split('\n')
    dataset_raw = [json.loads(case) for case in cases]
    dataset_raw_ch = dataset_raw[:800000]
    pairs_raw_ch = sorted(set([(it['problem'], it['solution']) for it in dataset_raw_ch]))
    print('len(dataset_raw):', len(dataset_raw))                  # => 809993
    print('len(dataset_raw_ch):', len(dataset_raw_ch))            # => 800000
    print('len(pairs_raw_ch) after dedup:', len(pairs_raw_ch))    # => 538949
    #dataset_raw_en = dataset_raw[800000:]
    #pairs_raw_en = sorted(set([(it['problem'], it['solution']) for it in dataset_raw_en]))
    #print('len(dataset_raw_en):', len(dataset_raw_en))            # => 9993
    #print('len(pairs_raw_en) after dedup:', len(pairs_raw_en))    # => 9993
  return pairs_raw_ch


# 仅返回中文数据 (未去重)
def load_dataset_processed() -> Dict[str, Any]:
  with open(DATASET_PROC_FILE, 'r', encoding='utf-8') as fh:
    dataset_proc = json.load(fh)
    print('len(dataset_proc):', len(dataset_proc))
    dataset_proc_ch = dataset_proc[:800000]
    print('len(dataset_proc_ch):', len(dataset_proc_ch))
  return dataset_proc_ch


def load_testset() -> List[QA]:
  with open(DATASET_TEST_FILE, 'r', encoding='utf-8') as fh:
    cases = fh.read().strip().split('\n')
  samples = [json.loads(case) for case in cases]
  return [(it['problem'], it['solution']) for it in samples]
