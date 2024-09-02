主要方法：

调优prefill batch size、decoding batch size 与请求发送间隔。

超参配置介绍：

```yaml
model_config:
    model_name: 'llama_7b'
    max_generate_length: 4096
    end_token: 2
    seq_length: [4096]
    vocab_size: 32000
    prefill_batch_size: [10]
    decode_batch_size: [50]
    zactivate_len: [512, 1024, 2048, 4096]
    model_type: 'dyn'
    seq_type: 'static'
    batch_waiting_time: 0.0
    decode_batch_waiting_time: 0.0
    batching_strategy: 'continuous'
    current_index: False
    page_attention: True
    model_dtype: "DataType.FLOAT32"
    pad_token_id: 0
    backend: 'kbk' # 'ge'
    model_cfg_path: '/home/ma-user/work/mindformers/configs/llama2/predict_llama2_7b.yaml'

serving_config:
    agent_ports: [16002]
    start_device_id: 0
    server_ip: '127.0.0.1'
    server_port: 8835

pa_config:
    num_blocks: 1024
    block_size: 16
    decode_seq_length: 4096

tokenizer:
    type: LlamaTokenizer
    vocab_file: '/home/ma-user/work/checkpoint_download/llama2/tokenizer.model'

basic_inputs:
    type: LlamaBasicInputs

extra_inputs:
    type: LlamaExtraInputs

warmup_inputs:
    type: LlamaWarmupInputs

```

优化后的推理总时长:

800.0236968994141

无额外运行环境

配置命令：
```
python test_serving_performance.py -X 3 -P 8835 -O "./" -T 500
```

运行日志：test_performance_2024-07-22-17_37

配置文件：llama_7b_kbk_pa_dyn.yaml

Llm-serving obs 路径：
https://tuili.obs.cn-southwest-2.myhuaweicloud.com/llm-serving.zip
