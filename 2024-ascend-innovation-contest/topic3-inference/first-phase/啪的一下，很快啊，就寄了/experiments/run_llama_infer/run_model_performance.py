#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/07/12 

# 直接加载模型来推理，跑整个测试数据集

import json
from time import time
from argparse import ArgumentParser
from typing import Union

import mindspore as ms
ms.set_context(mode=ms.GRAPH_MODE, device_target='Ascend', device_id=0)
from mindformers import LlamaTokenizer, LlamaTokenizerFast
from mindformers import MindFormerConfig, LlamaConfig, LlamaForCausalLM

# use relative path ↓↓↓
TOKENIZER_V2_PATH = './checkpoint_download/llama2/tokenizer.model'
CKPT_V2_PATH = './checkpoint_download/llama2/llama2_7b.ckpt'
CONFIG_V2_PATH = './mindformers/configs/llama2/predict_llama2_7b.yaml'
DATASET_PATH = './performance_serving/alpaca_5010.json'


''' CmdArgs '''
parser = ArgumentParser()
parser.add_argument('-C', '--cfg_file', help='path to predict_llama_*.yaml')
parser.add_argument('-F', '--vocab_file', help='path to tokenizer.model')
parser.add_argument('--fast', action='store_true', help='use fast version of llama2 tokenizer')
parser.add_argument('-M', '--ckpt_file', help='path to llama_*.ckpt checkpoint')
parser.add_argument('--seq_length', default=512, type=int, help='seq_length')
args = parser.parse_args()


''' Data '''
with open(DATASET_PATH, encoding='utf-8') as fh:
  alpaca_data = json.load(fh)
CASES = []
for data in alpaca_data[:1500]:
  txt = data["instruction"] + ":" + data["input"] if data["input"] else data["instruction"]
  CASES.append((txt, data["output"]))


''' Tokenzier '''
tokenizer: Union[LlamaTokenizer, LlamaTokenizerFast]
fp = args.vocab_file or TOKENIZER_V2_PATH
tokenizer = (LlamaTokenizerFast if args.fast else LlamaTokenizer)(fp)


''' Model '''
fp = args.cfg_file or CONFIG_V2_PATH
config = MindFormerConfig(fp)
model_config = LlamaConfig(**config.model.model_config)
model_config.seq_length = args.seq_length           # = total_workspace_len = input_len + max_ouput_len
model = LlamaForCausalLM(model_config)

# warm up
_ = model.generate(tokenizer.encode('Test'), max_new_tokens=1, do_sample=False)


''' Main Loop '''
predicts = []
S = time()
for X, Y in CASES:
  max_new_tokens = len(tokenizer.tokenize(Y))
  s = time()
  inputs = tokenizer.encode(X)
  outputs = model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=False)[0]
  generated = tokenizer.decode(outputs)
  t = time()
  print(f'>> time cost: {t - s:.5f}s')
  predicts.append({
    'inputs': X,
    'inputs_len': len(inputs),
    'outputs': generated,
    'outputs_len': len(tokenizer.tokenize(generated)),
    'time': t - s,
  })
T = time()
print(f'>> total time: {T - S:.5f}s')   # 2457.06609s


''' Result '''
with open('./run_model_performance.json', 'w', encoding='utf-8') as fh:
  json.dump(predicts, fh, indent=2, ensure_ascii=False)
