# 推理调优赛题carryhjr队伍提交

## 本作品使用的推理优化算法介绍

考虑到logits 误差必须小于0.005，无法使用kvcache环节的量化等模型方面常用的优化手段(均会
导致logits变动)，因此只能从调度层面出发, 基准baseline是采用-X 0.5 -T 3000 对应3551s,
显然可以提高发送频率来提高速度，但过快的发送可能导致 first token time 超过600s, 进而导致失
去响应, 同时也可以通过组batch方式提高推理速度，因此本作品主要从三个点进行优化:

1. 提高发送频率 -X 50 -T 30

2. 加入 interval-relax 机制, 这里 interval 设置1000, relax 设置5， 即发送1000个请求后 停顿
5s, 即可缓解失去响应的情况

3. 组batch, 在采用llm_server提供的 batch_continuous 策略同时，将prefill的batch和decode
的batch均设置为32, 即可有效提高推理速度

### 超参配置介绍

测速度时采用 llm-serving/configs/llama/llama_7b_kbk_pa_dyn_batch32.yaml 文
件 核心参数如下:

```
prefill_batch_size: [32]
decode_batch_size: [32]
batching_strategy: 'continuous'

```

测精度时采用 llm-serving/configs/llama/llama_7b_kbk_pa_dyn.yaml 变动参数如下:
```
prefill_batch_size: [1]
decode_batch_size: [1]
```

### 优化后的推理总时长

可见 test_performance_2024-07-31-00_05.log 总时长 846s

### 额外配置命令 

无 不过提供了一个简单的 kill_server.py 方便输出 server的 PID 不需要去挨个手动复制

## 运行命令和对应的提交推理的日志、配置文件

测速：

```
# server
python examples/start.py
configs/llama/llama_7b_kbk_pa_dyn_batch32.yaml
# client
python test_serving_performance.py -X 50 -P 8835 -O "./" -T 30
```

测速配置文件为 llm-serving/configs/llama/llama_7b_kbk_pa_dyn_batch32.yaml
测速日志为 test_performance_2024-07-31-00_05.log








