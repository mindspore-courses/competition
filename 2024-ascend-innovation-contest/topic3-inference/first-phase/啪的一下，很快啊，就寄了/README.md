# 推理调优实验说明

----

完整的实验仓库下载链接: https://vhaktyr.obs.cn-southwest-2.myhuaweicloud.com/work.zip

⚠ 解压覆盖到 `/home/ma-user/work`，然后运行 `source ./init.sh` 以搭建环境


### Quickstart

⚪ 速度测试

实验结果见日志: [test_performance_2024-07-31-20_13.log](./test_performance_2024-07-31-20_13.log)，相比基线约提升 `26.65%` (Exec Time: 2605.3589s)

```shell
# launch llm-serving
cd /home/ma-user/work/llm-serving/
python examples/start.py --task 1 --config configs/llama/llama_7b_kbk_pa_dyn.yaml

# test llm-serving (warm up)
# 一定要预热，不然测不准！
curl 127.0.0.1:8835/models/llama2/generate \
  -X POST \
  -d '{"inputs":" I love Beijing, because","parameters":{"max_new_tokens":10, "do_sample":"True", "return_full_text":"True"}, "stream":"True"}' \
  -H 'Content-Type: application/json'

# run performance_serving task-1
cd /home/ma-user/work/performance_serving/
python test_serving_performance.py --task 1 -X 0.625 -T 2400
```

⚪ 精度测试

精度测试脚本如下：

```shell
# launch llm-serving
cd /home/ma-user/work/llm-serving/
python examples/start.py --task 2 --config configs/llama/llama_7b_kbk_pa_dyn.yaml

# run performance_serving task-2
cd /home/ma-user/work/performance_serving/
# 这是规定的标准配置 ↓↓↓
python test_serving_performance.py --task 2 -X 0.1 -T 5000
# 如果觉得标准配置很慢，也可以用下面的命令 ↓↓↓
python test_serving_performance.py --task 2 -X 0.4 -T 1250
# test precision
cd /home/ma-user/work/
python acc_allclose.py --base_path file_npy_base --new_path file_npy
```


### Explanations

服务侧 (llm-serving) 的改动: 删除了影响性能的 `logging.debug`，`np.concatenate` 改为 `np.pad`
模型侧 (mindformers) 的改动: 仅配置文件修改 `model_length=512`

原项目时间打点测试如下：

```
时间开销比例 [service / model = 1/2 ~ 3/5]
[performance]  3513.11s = 58.55min (推算 35.81ms/token， 多线程测定总时间)
[llm-serving] ~3996.02s = 66.60min (测定 40ms/token, 单线程推算总时间)
[mindformers]  2395.26s = 39.92min (测定 25ms/token, 单线程测定总时间)
 (overhead)               18.63min
```

假设模型侧的 `25ms/token` 速度已经无法通过简单手段进行优化，故核心问题在于**定位服务侧的 15ms/token 的额外开销**！

```
[mindformers call-chain]
-> GenerationMixin.generate                   // 循环推理，测定 25ms/token
  -> GenerationMixin.infer                    // 推理一个token单位
    -> GenerationMixin.forward (*)            // 推测 ~22.5ms
      -> LlamaForCausalLM.construct           // if prefill
      -> GenerationMixin._incremental_infer   // if decode
        -> LlamaForCausalLM.construct
    -> GenerationMixin.postprocess            // 推测 ~2.5ms

[llm-serving call-chain]
-> LLMServer.generate_answer                              // 推理一个请求单位
  -> LLMServer.register_request
    -> LLMServer.start_background_loop
      -> LLMServer.run_loop
        -> LLMServer.step
          -> AsyncMaster.step_async                       // 决定 batching 策略
            -> AsyncMaster._run_workers_async             // 测定 40ms/token -> 优化后 27ms/token
              -> Worker.predict                           // 测定 39.55ms
                - Worker._predict
                  -> DisModel.call                        // 测定 39.47ms
                    - shared_mem::write
                    - tcp::sendall
                      -> start_agent_socket_server        // 测定 63.4ms(prefill) / 37.7ms(decode)
                        - tcp::recv
                        - WorkAgent.predictc
                          - shared_mem::read
                          - agent pre-process             // 测定 12.0ms [这tm就是额外开销!!]
                          - WorkAgent.predict_for_kbk     // 测定 22.5ms
                            -> GenerationMixin.forward    // 流程同上 mindformers (*)
                          - WorkAgent.do_post_sampling    // 测定 3.04ms [这tm也有一点额外开销!!]
                          - shared_mem::write
                        - tcp::sendall
                    - tcp::recv
                    - shared_mem::read
              - AsyncMaster._postprocess                  // 测定 0.83ms
```

↑↑↑ 额外开销分析 & 解决 ↓↓↓

```
[WorkAgent.predict (agent pre-process)]
- preprocess_time: 12.47ms
  - logging.debug: ~7ms (移除!)
  - np.concatenate: 4.8ms (改用np.pad!)
- predict_time: 22.48ms
- post_time: 3.59ms
  - logits_to_token: 2.522ms
  - write_shared_mem: 0.716ms
```

排查函数调用链后去掉上述两个最大的性能影响因素，单token推理时间从 `40ms/token` 降至 `27ms/token`，已经逼近裸的 mindformers 速度 `25ms/token`，
其中仍存在的 `2ms/token` 额外开销用于 TCP 和 SharedMem 通信，代码过于细碎无法优化了。

推理过程中 AI Core 占用率保持 60%，或仍存在优化空间；由于时间因素，模型侧优化留待以后继续讨论... ;)

----
by Armit
2024/07/31
