# 1. 业界推理优化算法调研

- 1. 模型压缩
量化：将模型的参数从高精度（如 32 位浮点数）转换为低精度（如 8 位整数），减少模型的存储和计算量。例如，可以使用 INT8 量化来大幅降低计算开销。
剪枝：去除模型中不重要的参数或连接，减少模型的规模。比如对于某些神经元连接，如果其对最终输出的影响较小，可以将其剪除。
- 2. 硬件优化
使用专用的硬件加速设备，如 GPU、TPU 等。例如，NVIDIA 的 A100 GPU 在处理大规模模型计算时具有出色的性能。
优化硬件的配置和设置，如合理分配内存、调整缓存大小等。
- 3. 模型并行化
数据并行：将数据分布到多个计算节点上，同时进行计算。
模型并行：将模型拆分成多个部分，分布在不同的计算节点上并行计算。
- 4. 优化算法和库
使用高效的深度学习框架和优化算法，例如 TensorFlow、PyTorch 等框架都提供了各种优化选项。
利用现有的优化库，如 cuDNN 等，加速特定的计算操作。
- 5. 缓存和预取
对经常使用的数据或计算结果进行缓存，避免重复计算。
预取可能需要的数据，减少数据等待时间。
- 6. 模型架构优化
选择更高效的模型架构，例如使用轻量级的模型结构。
减少模型的层数或参数数量，在保证性能的前提下降低计算复杂度。
- 7. 混合精度计算
结合不同精度的计算，在关键部分使用高精度，在非关键部分使用低精度。
例如，在一些图像识别任务中，通过对模型进行量化和剪枝，在不明显降低准确率的情况下，推理速度可以提高数倍。又如，一些大型互联网公司通过部署大规模的 GPU 集群，并对模型进行并行化处理，实现了对用户请求的快速响应，大幅提升了服务质量。

# 2. 本作品使用的推理优化算法介绍

硬件优化，调整内存分配和缓存大小

# 3. 超参配置介绍

```yaml
model_config:
    model_name: 'llama_7b'
    max_generate_length: 4096
    end_token: 2
    seq_length: [4096]
    vocab_size: 32000
    prefill_batch_size: [32]
    decode_batch_size: [32]
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

# 4. 优化后的推理总时长

X 0.8 
T 1875

推理总时长：1888.1983 s

# 5. 运行环境说明

无额外配置
