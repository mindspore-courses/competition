# 实验

    如无特殊说明，这些实验应该顺序进行

----

### 云端实验

- 基线复现: [baseline_reproduce](./baseline_reproduce/README.md)
- 测试裸 mindformer: [run_llama_infer](./run_llama_infer/README.md)
- 测试 llm-serving 套壳: [run_llm_serving](./run_llm_serving/README.md)
- 测试 run_performance_serving: 套壳: [run_performance_serving](./run_performance_serving/README.md)

```
时间开销比例 [service / model = 1/2 ~ 3/5]
[performance]  3513.11s = 58.55min (推算 35.81ms/token， 多线程测定总时间)
[llm-serving] ~3996.02s = 66.60min (测定 40ms/token, 单线程推算总时间)
[mindformers]  2395.26s = 39.92min (测定 25ms/token)
 (overhead)               18.63min
```

⚠ 核心问题在于定位 15ms/token 的额外开销！

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


### 本地实验

- 数据集统计: [ds_stats](./ds_stats/README.md)
- 裸跑测试 / 打桩测试: [bare_run](./bare_run/README.md)
- lprof 测试: [lprof](./lprof/README.md)
