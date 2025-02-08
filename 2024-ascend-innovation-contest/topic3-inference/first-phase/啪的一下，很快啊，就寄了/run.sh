#!/usr/bin/env bash

# launch llm-serving (~5min)
# see logs under /home/ma-user/work/llm-serving, agent.log and server.log
# occasionally fails due to OOM, just try again :(
cd /home/ma-user/work/llm-serving/
python examples/start.py --task 1 --config /home/ma-user/work/llm-serving/configs/llama/llama_7b_kbk_pa_dyn.yaml

python examples/start_agent.py --task 1 --config configs/llama/llama_7b_kbk_pa_dyn.yaml
python examples/server_app_post.py --config configs/llama/llama_7b_kbk_pa_dyn.yaml


# test llm-serving (warm up)
# 一定要预热，不然测不准！
curl 127.0.0.1:8835/models/llama2/generate \
  -X POST \
  -d '{"inputs":" I love Beijing, because","parameters":{"max_new_tokens":56, "do_sample":"True", "return_full_text":"True"}, "stream":"True"}' \
  -H 'Content-Type: application/json'


# run performance_serving
cd /home/ma-user/work/performance_serving/

# test task 1 (--task should match with examples/start.py)
python test_serving_performance.py --task 1 -X 0.625 -T 2400

# test task 2 (--task should match with examples/start.py)
python test_serving_performance.py --task 2 -X 0.1 -T 5000


# test precision
cd /home/ma-user/work/
python acc_allclose.py --base_path file_npy_base --new_path file_npy
