# 基线复现实验

----

### Task 1

⚪ 云端

> [time cost] 3530.0440 (record) / 3548 (wall)

```bash
cd /home/ma-user/work/llm-serving/
python examples/start.py --task 1 --config /home/ma-user/work/llm-serving/configs/llama/llama_7b_kbk_pa_dyn.yaml

cd /home/ma-user/work/performance_serving/
# 改 test.sh 里的主要命令行为 ↓
# python test_serving_performance.py --task 1 -X 0.5 -T 3000
nohup sh test.sh > task.log 2>&1 &
watch -n 10 tail task.log
```

⚪ 本地

```bat
CD "%WORK%\llm-serving"
python examples/start_agent.py --task 1 --config configs\llama\llama_7b_kbk_pa_dyn_debug.yaml
python examples/server_app_post.py --config configs\llama\llama_7b_kbk_pa_dyn_debug.yaml

REM 前 50 个样例 (可以跑更少)
CD "%WORK%\performance_serving"
python test_serving_performance.py --task 1 -X 1 -T 50
```


### Task 2

> [time cost] 5008.9543 (record) / 5026 (wall), precision passed

⚪ 云端

```bash
cd /home/ma-user/work/llm-serving/
python examples/start.py --task 2 --config /home/ma-user/work/llm-serving/configs/llama/llama_7b_kbk_pa_dyn.yaml

cd /home/ma-user/work/performance_serving/
# 改 test.sh 里的主要命令行为 ↓
# python test_serving_performance.py --task 2 -X 0.1 -T 5000
nohup sh test.sh > task.log 2>&1 &
watch -n 10 tail task.log

# test precision
cd /home/ma-user/work/
python acc_allclose.py --base_path file_npy_base --new_path file_npy
```

⚪ 本地

```bat
CD "%WORK%\llm-serving"
python examples/start_agent.py --task 2 --config configs\llama\llama_7b_kbk_pa_dyn_debug.yaml
python examples/server_app_post.py --config configs\llama\llama_7b_kbk_pa_dyn_debug.yaml

REM 前 50 个样例，能产生个 npy 文件
CD "%WORK%\performance_serving"
python test_serving_performance.py --task 2 -X 0.2 -T 250

CD "%WORK%"
python acc_allclose.py --base_path file_npy_base --new_path file_npy
```
