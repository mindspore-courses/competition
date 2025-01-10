#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/07/31 

# 验证是否调度中存在气泡：假定单 token 推理时间固定，模拟计算给定请求序列的最小耗时
# 结论: 气泡时间不到 2%，根本原因还是序列太长，模型推理推理时间瓶颈

import json
from mindformers import LlamaTokenizer

tokenizer = LlamaTokenizer("../../performance_serving/tokenizer.model")
with open("../../performance_serving/alpaca_5010.json", "r", encoding="utf-8") as fh:
  alpaca_data = json.load(fh)[:1500]

output_token_lengths = []
for data in alpaca_data:
  input_ = data["instruction"] + ":" + data["input"] if data["input"] else data["instruction"]
  output_token_lengths.append(len(tokenizer.tokenize(data["output"])))


def estimate_ts(rts:float, qps:float):
  spq = 1 / qps
  ts_total = 0
  ts_infer = 0
  ts_bubble = 0
  ts_wait_max = 0
  for i, length in enumerate(output_token_lengths):
    sample_ts = length * rts
    ts_infer += sample_ts
    if spq > sample_ts:
      ts_bubble += spq - sample_ts
    ts_expect = i * spq
    ts_total = max(ts_total + sample_ts, ts_expect)
    ts_wait_max = max(ts_wait_max, ts_total - ts_expect)

  print('infer    time:', ts_infer)
  print('bubble   time:', ts_bubble)
  print('total    time:', ts_total)
  print('max wait time:', ts_wait_max)
  print()


# 依 llm-serving
# infer    time: 3923.44
# bubble   time:  966.44
# total    time: 4889.88
# max wait time: 1891.88
estimate_ts(0.04, 0.5)

# 依 mindformers
# infer    time: 2452.15
# bubble   time: 1260.80
# total    time: 3712.95
# max wait time:  714.95
estimate_ts(0.025, 0.5)


''' 依优化后 '''
# infer    time: 2648.3219999999997
# bubble   time: 1208.6099999999988
# total    time: 3021.075
# max wait time: 50.64499999999953
estimate_ts(0.027, 0.5)
# infer    time: 2648.3219999999997
# bubble   time: 854.2609999999992
# total    time: 2654.1149999999993
# max wait time: 255.71499999999924
estimate_ts(0.027, 0.625)
# infer    time: 2648.3219999999997
# bubble   time: 639.3636666666661
# total    time: 2652.5149999999994
# max wait time: 653.8483333333329
estimate_ts(0.027, 0.75)
# infer    time: 2648.3219999999997
# bubble   time: 397.36100000000056
# total    time: 2650.5149999999994
# max wait time: 1151.5149999999994
estimate_ts(0.027, 1.0)
# infer    time: 2648.322
# bubble   time: 191.7113
# total    time: 2648.523
# max wait time: 1649.19
estimate_ts(0.027, 1.5)
# infer    time: 2648.322
# bubble   time: 110.095
# total    time: 2648.322
# max wait time: 1898.822
estimate_ts(0.027, 2.0)
