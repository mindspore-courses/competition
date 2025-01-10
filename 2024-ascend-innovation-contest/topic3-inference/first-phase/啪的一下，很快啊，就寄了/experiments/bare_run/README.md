# Bare Run Tests 裸跑测试

```txt
由于完整的调用链很长:
  performance_serving -[http]-> llm-serving
    -> server_app_post -> llm_server -> request_engine (queue)
      -> worker -[tcp+shared_mem]-> agent
        -> mindformer_model
我们需要定位哪个部分的耗时是瓶颈，或者可优化的 :)

设置环境变量 RUN_LEVEL=<lv> 来测试下列等级，观察 test_serving_performance.py 测试脚本给出的端到端总时长 Exec Time
  Level 1: FAKE_SERVER_APP_POST   数据到达 server_app_post.async_stream_generator 后返回
  Level 2: FAKE_WORKER            数据到达 Worker.predict 后返回
  Level 3: FAKE_DISPATCHER        数据到达 DisModel.call 后返回
  Level 4: FAKE_AGENT             数据到达 WorkAgent.predict 后返回
  Level 5: FAKE_AGENT_CALL_MODEL  数据到达 WorkAgent.predict_for_kbk 后返回   <- 默认等级
  Level 6: FAKE_MODEL_FORWARD     数据到达 LlamaModel.construct 后返回        <- 本地 CPU 环境能跑
  Level 99: REAL_MODEL_FORWARD    真实模型推理过程 (无预训练权重)              <- 本地 CPU 环境跑不起来
  Level 100: REAL_CKPT_FORWARD    真实模型推理过程 (加载预训练权重)
```

```bat
REM 两个服务器都要设置这个变量
SET RUN_LEVEL=3

CD "%WORK%\llm-serving"
python examples/start_agent.py --task 1 --config configs\llama\llama_7b_kbk_pa_dyn_debug.yaml
python examples/server_app_post.py --config configs\llama\llama_7b_kbk_pa_dyn_debug.yaml

CD "%WORK%\performance_serving"

python test_serving_performance.py --task 1 -X 1 -T 30
```

⚠ 结论: 核心开销在RUN_LEVEL 4 ~ 5 之间, 有 ~10s 延迟 (定位为 `WorkAgent.predict` 方法)

| Level | Time | Cmd | max/last `[generate_answer]` |
| :-: | :-: | :-: | :-: |
| 1 | 30.9921 | -X 1 -T 30 | |
| 2 | 31.4550 | -X 1 -T 30 | |
| 3 | 31.2791 | -X 1 -T 30 |  0.0    |
| 4 | 31.4567 | -X 1 -T 30 |  0.1357 |
| 5 | 32.1628 | -X 1 -T 30 |  0.7378 |
| 6 | 32.1843 | -X 1 -T 30 |  0.7694 |
| 1 | 11.7557 | -X 3 -T 10 | |
| 2 | 12.0357 | -X 3 -T 10 | |
| 3 | 11.9752 | -X 3 -T 10 |  0.0    |
| 4 | 11.8221 | -X 3 -T 10 |  0.1245 |
| 5 | 20.1493 | -X 3 -T 10 |  8.4359 |
| 6 | 23.9568 | -X 3 -T 10 | 11.9208 |
| 1 |  5.1224 | -X 10 -T 3 | |
| 2 |  5.2403 | -X 10 -T 3 | |
| 3 |  5.2668 | -X 10 -T 3 |  0.087  |
| 4 |  5.7206 | -X 10 -T 3 |  0.8585 |
| 5 | 22.7302 | -X 10 -T 3 | 17.5434 |
| 6 | 19.8541 | -X 10 -T 3 | 14.6366 |
| 1 |  3.0659 | -X 30 -T 1 | |
| 2 |  4.1349 | -X 30 -T 1 | |
| 3 |  4.3945 | -X 30 -T 1 |  1.36   |
| 4 |  5.1491 | -X 30 -T 1 |  2.0258 |
| 5 | 25.9519 | -X 30 -T 1 | 22.7519 |
| 5 | 15.6472 | -X 30 -T 1 | 12.5312 (数据不太稳定，不知道为何) |
| 6 | 16.5957 | -X 30 -T 1 | 13.4657 |


#### `WorkAgent.predict` 分析

```
npu_total_time = 40.42506217956543
  - pre-process  time = 13.292789459228516
  - kbk predict  time = 23.569345474243164
  - post process time =  3.093719482421875
```
