## 推理优化赛道

队伍名：hack_ai2

代码地址：
https://zyf-llm-code.obs.cn-southwest-2.myhuaweicloud.com/2024_ascend/ms_inference/code.zip

npy文件地址（这些推理生成的文件其实和baseline一模一样，我用md5对比过）：

https://zyf-llm-code.obs.cn-southwest-2.myhuaweicloud.com/2024_ascend/ms_inference/file_npy.zip

1500条数据推理时间：789.7秒，精度测试通过。


以上数据启动和复现的操作过程与文档一致，下面简述下修改的地方。

1.	llm-serving中的agent_multi_post_method.py文件，设置算子并行数为8，以及开启图算融合，这样能稍微提升一点推理速度(此修改对精度不会有任何影响)：

```
mindspore.set _context(inter_op_parallel_num=8)
mindspore.set_context(enable_graph_kernel=True)
```
2.	Llm-serving中的llama_7b_kbk_pa_dyn.yaml配置文件，测试推理时间时把decode_batch_size改成32，进行多batch推理，但测试精度时需要改回为1，因为多batch推理时，每条数据的推理顺序不能保证与原来一样，会影响精度的评测


3.	performance_serving中的test.sh脚本，把-X设置为16，-T设置为94进行推理时间测试（这样一共推理了1504条，目前测下来X=16效果最好，但不能整除1500，所以只能多推理4条），进行精度测试时，把-X设置为1，-T设置为500

4.	到llm-serving中执行以下命令启动服务：

```
python examples/start.py --config /home/ma-user/work/llm-serving/configs/llama/llama_7b_kbk_pa_dyn.yaml
```


服务启动后，如果是测试时间，先用以下命令进行一次预推理：
```
curl 127.0.0.1:8835/models/llama2/generate -X POST -d '{"inputs":" I love Beijing, because","parameters":{"max_new_tokens":16, "do _sample":"True", "return_full_text":"True"}, "stream":"True"}' -H 'Content-Type: application/json'
```

因为第一次推理速度比较慢，可能需要30秒左右，然后在performance_serving中运行以下脚本进行测试速度
```
nohup sh test.sh > test_sh.log 2>&1 &
```

如果是测试精度，不需要预推理，先确保llm-serving中的agent_multi_post_method_save_logits.py已经替换了agent_multi_post_method.py，以及performance_serving中的test_serving_performance.py里的数据集改成了alpaca_521.json，然后直接在performance_serving中执行sh test.sh，测试精度的时间比较长，要将近55分钟

