#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/07/12 

# 直接加载模型来推理，交互式测试玩玩

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


''' CmdArgs '''
parser = ArgumentParser()
parser.add_argument('-C', '--cfg_file', help='path to predict_llama_*.yaml')
parser.add_argument('-F', '--vocab_file', help='path to tokenizer.model')
parser.add_argument('--fast', action='store_true', help='use fast version of llama2 tokenizer')
parser.add_argument('-M', '--ckpt_file', help='path to llama_*.ckpt checkpoint')
parser.add_argument('--skip_load', action='store_true', help='skip load llama_*.ckpt')
parser.add_argument('--seq_length', default=128, type=int, help='seq_length')
parser.add_argument('--max_length', default=64, type=int, help='max length for predict output')
args = parser.parse_args()


''' Tokenzier '''
tokenizer: Union[LlamaTokenizer, LlamaTokenizerFast]
fp = args.vocab_file or TOKENIZER_V2_PATH
tokenizer = (LlamaTokenizerFast if args.fast else LlamaTokenizer)(fp)


''' Model '''
fp = args.cfg_file or CONFIG_V2_PATH
config = MindFormerConfig(fp)
model_config = LlamaConfig(**config.model.model_config)
model_config.seq_length = args.seq_length           # = total_workspace_len = input_len + max_ouput_len
model_config.max_decode_length = args.max_length    # = max_ouput_len
if args.skip_load:
  model_config.checkpoint_name_or_path = None
model = LlamaForCausalLM(model_config)


''' Main Loop '''
try:
  while True:
    txt = input('>> input your sentence: ').strip()
    if not txt: continue
    inputs = tokenizer.encode(txt)
    print(f'>> input_ids({len(inputs)}): {inputs}')
    s = time()
    outputs = model.generate(inputs, max_new_tokens=model_config.max_decode_length, do_sample=False)[0]
    t = time()
    print(f'>> output_ids({len(outputs)}): {outputs}')
    generated = tokenizer.decode(outputs)
    print(f'>> generated({len(generated)}): {generated}')
    print(f'>> time cost: {t - s:.5f}s')
except KeyboardInterrupt:
  print('[Exit by Ctrl+C]')
