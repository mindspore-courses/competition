#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/06/29 

# 算子速度测试: 随机输入跑随机初始化的模型

import os
import sys
import json
from time import time
from tqdm import tqdm
import mindspore as ms
import mindspore.nn as nn ; nn.Cell
import mindspore.ops.functional as F
from mindformers import AutoModel, LlamaTokenizer, LlamaForCausalLM

IS_WIN = sys.platform == 'win32'
os.environ['RUN_LEVEL'] = '99'
ms.set_seed(114514)

# use relpath, work around for path with whitespaces...
BASE_PATH = '../../'
TEST_FILE = BASE_PATH + 'performance_serving/alpaca_521.json'
if IS_WIN:
  ms.set_context(mode=ms.PYNATIVE_MODE, device_id=0, device_target='CPU', pynative_synchronize=True)
  LLAMA_CONFIG_PATH = BASE_PATH + 'mindformers/configs/llama2/predict_llama2_7b_debug.yaml'
  TOKENIZER_PATH = BASE_PATH + 'performance_serving/tokenizer.model'
else:
  ms.set_context(mode=ms.GRAPH_MODE, device_id=0, device_target='Ascend')
  LLAMA_CONFIG_PATH = BASE_PATH + 'mindformers/configs/llama2/predict_llama2_7b.yaml'
  TOKENIZER_PATH = BASE_PATH + 'checkpoint_download/llama2/tokenizer.model'


# it takes ~50s to launch on CPU in PY_NATIVE mode
tokenizer = LlamaTokenizer(TOKENIZER_PATH)
model: LlamaForCausalLM = AutoModel.from_config(LLAMA_CONFIG_PATH, download_checkpoint=False)
model.set_train(False)


#@profile
def benchmark():
  # warm up
  TEST_SAMPLE = 'This is a simple bare test that runs for LLAMA model performance profiling :)'
  inputs = tokenizer([TEST_SAMPLE], return_tensors='ms')
  logits, tokens, mask = model(inputs['input_ids'])
  preds = F.argmax(logits, -1)
  assert preds is not None
  if not IS_WIN: return     # 傻逼 Ascend 只能跑起来一个

  # benchmark
  N_SAMPLES = 5
  with open(TEST_FILE, 'r', encoding='utf-8') as fh:
    data = json.load(fh)[:N_SAMPLES]
    TEST_SAMPLES = [it["instruction"] + it['input'] for it in data]

  for i, txt in enumerate(tqdm(TEST_SAMPLES)):
    print(f'[{i}] {txt}')
    inputs = tokenizer([txt], return_tensors='ms')
    logits, tokens, mask = model(inputs['input_ids'])
    preds = F.argmax(logits, -1)
    assert preds is not None

ts = time()
benchmark()
print('time cost:', time() - ts)


#from code import interact
#interact(local=globals())
