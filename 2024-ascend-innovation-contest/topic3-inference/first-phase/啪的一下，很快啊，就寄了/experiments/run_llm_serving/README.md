### llm-serving 时间打点

ℹ 使用 curl 测试，非 sse 模式

⚪ 服务侧对内调用端到端时间 (秒s)

| output length | `send_request` | `LLMServer.generate_answer` | `get_full_res` | time per token | comment |
| :-: | :-: | :-: | :-: | :-: | :-: |
|   1 | 0.0001990795135498047  |  0.06877923011779785 |  0.06897521018981934 | 0.068975210189819 | |
|   5 | 0.00018739700317382812 |  0.23348283767700195 |  0.23371005058288574 | 0.046742010116577 | |
|  10 | 0.00021958351135253906 |  0.454270601272583   |  0.4545447826385498  | 0.045454478263855 | |
|  30 | 0.00020813941955566406 |  1.2642827033996582  |  1.2646539211273193  | 0.042155130704244 | |
|  60 | 0.00024700164794921875 |  2.4974281787872314  |  2.4978628158569336  | 0.041631046930949 | |
|  65 | 0.0002522468566894531  |  2.7048661708831787  |  2.7053494453430176  | 0.041620760697585 | avg len |
| 120 | 0.00019788742065429688 |  5.602715492248535   |  5.603420734405518   | 0.046695172786713 | |
| 240 | 0.00025391578674316406 | 10.093211650848389   | 10.09423828125       | 0.042059326171875 | |
| 360 | 0.00026154518127441406 | 15.20837664604187    | 15.20983362197876    | 0.04224953783883  | |
| 480 | 0.00025200843811035156 | 20.217636108398438   | 20.21990203857422    | 0.042124795913696 | |

⚪ 模型侧单 token 端到端推理时间 (毫秒ms)

```
[AsyncMaster._run_workers_async -> Worker._predict -> DisModel.call -> WorkAgent.predict]
  /home/ma-user/work/llm-serving/mindspore_serving/worker/worker.py[line:127] - INFO: pre-process time is 0.08249282836914062 
    /home/ma-user/work/llm-serving/mindspore_serving/worker/model_init_multimodel.py[line:505] - DEBUG: get input lite time is 0.1933574676513672 
      /home/ma-user/work/llm-serving/mindspore_serving/agent/agent_multi_post_method_task1.py[line:756] - INFO: agent pre-process time is 11.926651000976562
      /home/ma-user/work/llm-serving/mindspore_serving/agent/agent_multi_post_method_task1.py[line:820] - INFO: predict time is 22.455215454101562
      /home/ma-user/work/llm-serving/mindspore_serving/agent/agent_multi_post_method_task1.py[line:766] - INFO: multi_thread_post_sampling time is 3.0431747436523438
      /home/ma-user/work/llm-serving/mindspore_serving/agent/agent_multi_post_method_task1.py[line:768] - INFO: post_time is 3.2088756561279297
      /home/ma-user/work/llm-serving/mindspore_serving/agent/agent_multi_post_method_task1.py[line:769] - INFO: npu_total_time is 38.44714164733887
    /home/ma-user/work/llm-serving/mindspore_serving/worker/model_init_multimodel.py[line:530] - INFO: model.call time is 39.46876525878906 
  /home/ma-user/work/llm-serving/mindspore_serving/worker/worker.py[line:144] - INFO: DecodeTime 39.5505428314209
/home/ma-user/work/llm-serving/mindspore_serving/master/master.py[line:420] - INFO: post_process_time time is 0.3349781036376953
/home/ma-user/work/llm-serving/mindspore_serving/master/master.py[line:421] - INFO: e-to-e time is 40.23146629333496
```

↑↑↑ 上述数据分析 ↓↓↓

```
单 token 推理时间很稳定约 40ms，则上述两表的时间开销是能对齐的
    1*40/1000 =  0.04 ~=  0.06897521018981934 -> 72.438%
    5*40/1000 =  0.2  ~=  0.23371005058288574 -> 16.855%
   10*40/1000 =  0.4  ~=  0.4545447826385498  -> 13.636%
   30*40/1000 =  1.2  ~=  1.2646539211273193  ->  5.388%
   60*40/1000 =  2.4  ~=  2.4978628158569336  ->  4.078%
   65*40/1000 =  2.6  ~=  2.7053494453430176  ->  4.052%
  120*40/1000 =  4.8  ~=  5.603420734405518   -> 16.738%
  240*40/1000 =  9.6  ~= 10.09423828125       ->  5.148%
  360*40/1000 = 14.4  ~= 15.20983362197876    ->  5.624%
  480*40/1000 = 19.2  ~= 20.21990203857422    ->  5.312%

额外开销率有聊个比较典型的值：5% 和 16%，有无可能是内存回收导致的？
```
