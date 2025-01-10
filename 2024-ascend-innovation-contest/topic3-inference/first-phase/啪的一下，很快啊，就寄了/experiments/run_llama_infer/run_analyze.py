#!/usr/bin/env python3
# Author: Armit
# Create Time: 周五 2024/07/12 

import matplotlib.pyplot as plt
import json

with open('run_model_performance.json', encoding='utf-8') as fh:
  data = json.load(fh)

input_lens = [it['inputs_len'] for it in data]
output_lens = [it['outputs_len'] - it['inputs_len'] + 1 for it in data]
time = [it['time'] for it in data]

print('max(output_lens):', max(output_lens))  # 505
print('sum(time):', sum(time))  # 2395.26s = 39.92min

plt.figure(figsize=(8, 3))
plt.subplot(131) ; plt.title('len(Input) - Time')        ; plt.scatter(input_lens,  time,        s=5)
plt.subplot(132) ; plt.title('len(Input) - len(Output)') ; plt.scatter(input_lens,  output_lens, s=5)
plt.subplot(133) ; plt.title('len(Output) - Time')       ; plt.scatter(output_lens, time,        s=5)
plt.tight_layout()
plt.savefig('./run_analyze.png', dpi=300)
