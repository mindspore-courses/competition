## 业界推理优化算法调研

大模型推理优化算法主要是为了提高大型语言模型（如GPT-3、BERT等）在推理阶段的效率。以下是几种经典的优化算法：

1. Continuous Batching

Continuous Batching 是一种通过不断收集和批处理多个请求来提高推理效率的方法。

核心思想：将多个独立的推理请求收集到一个批次中，并进行一次性处理。这种方法利用了硬件的并行计算能力，提高了吞吐量。

实现方式：系统在接收到推理请求时不会立即执行，而是等待一段时间（如几毫秒）以收集更多的请求，然后一起进行批处理。

优点：提高了计算资源的利用率，减少了每个请求的平均处理时间。

缺点：可能会引入一定的延迟，特别是在请求到达率较低的情况下。

2. PagedAttention

PagedAttention 是一种优化自注意力机制（self-attention）的算法，旨在减少内存使用和计算复杂度。

核心思想：将自注意力机制中的大矩阵分解为多个小页（page），每次只处理一页数据，从而降低内存占用和计算量。

实现方式：在计算自注意力时，将输入分割成小块，逐块计算注意力分数和上下文表示，再组合结果。

优点：显著降低了自注意力机制的内存需求，适合在资源受限的环境中使用。

缺点：增加了实现复杂性，可能引入额外的计算开销。

3. Mixed Precision Training

Mixed Precision Training 是一种利用低精度浮点数（如FP16）进行计算以提高计算速度和减少内存使用的技术。

核心思想：在训练和推理过程中，部分计算使用低精度浮点数（FP16），而保持关键参数（如权重）使用高精度浮点数（FP32）。

实现方式：使用硬件支持混合精度计算的特性（如NVIDIA的Tensor Cores），通过自动混合精度（AMP）库简化实现。

优点：大幅提高计算速度和模型吞吐量，减少显存占用。

缺点：可能引入数值不稳定性，需要额外的技巧（如损失标度）来确保训练和推理的稳定性。


4. Model Quantization

Model Quantization 是一种通过将模型参数和计算从浮点数转换为更低位数整数（如INT8）的方法，以减少模型大小和加速推理。

核心思想：将浮点数表示的模型参数转换为低精度的整数表示，从而减少内存占用和计算复杂度。

实现方式：通过定点表示和量化感知训练（QAT）等技术，将模型权重和激活函数量化为低精度整数。

优点：显著减少了模型的存储需求和推理时间，适用于资源受限的设备。

缺点：可能会导致模型精度的下降，需要精心设计量化策略以最小化性能损失。

5. Knowledge Distillation

Knowledge Distillation 是一种通过训练一个小模型（学生模型）模仿一个大模型（教师模型）的行为来实现模型压缩和加速推理的方法。

核心思想：利用大模型的输出作为软标签（soft labels）来训练小模型，使小模型能够学习大模型的知识和行为。

实现方式：通过同时训练学生模型和教师模型，最小化学生模型输出和教师模型输出之间的差异。

优点：可以显著减少模型大小和推理时间，同时保持较高的精度。

缺点：需要额外的训练步骤和计算资源来生成软标签和训练学生模型。

## 本作品使用的推理优化算法介绍
我认为，想要提升1500条推理请求的耗时，最优的方式就是调整batching的策略，目前最优的组batching的策略就是continuous batching方法，观察了目前的llm-servering框架在开启pa以后，只支持单个batch的prefill阶段推理，但是支持decoding阶段的多batch推理，所以我的优化思路第一步就是：
使用continuous batching算法，将decoding阶段的batch数提升，并进行消融实验测试推理耗时效果比较，寻找最优的decode_batch_size参数。

此处省略消融实验的数据结果。

调整完batchsize后，再调整句子长度，在普通的KVcache存储，一般都是预先分配固定的显存空间用于存储，但是会造成极大的资源浪费。所以有了pagedattention算法，可以按照操作系统分页的思维，动态分配显存用于存放KVcache。

最终数据比较分析：
根据推理日志，将最初版的日志，与当前最优版本的日志进行对比，主要是对比frist_token_time和res_time这两个参数，一个是该请求的首个token生成延时，另一个是整个句子生成耗时。
分析代码如下：

```
import re
import matplotlib.pyplot as plt

# 假设日志文件名为 'logfile1.log' 和 'logfile2.log'
logfiles = ['logfile1.log', 'logfile2.log']

res_time_pattern = re.compile(r"'res_time': (\d+\.\d+)")
first_token_time_pattern = re.compile(r"'first_token_time': (\d+\.\d+)")

res_time_lists = [[], []]
first_token_time_lists = [[], []]
line_numbers = [[], []]

for idx, logfile in enumerate(logfiles):
    with open(logfile, 'r') as file:
        for i, line in enumerate(file, 1):
            res_time_match = res_time_pattern.search(line)
            first_token_time_match = first_token_time_pattern.search(line)

            if res_time_match and first_token_time_match:
                res_time_lists[idx].append(float(res_time_match.group(1)))
                first_token_time_lists[idx].append(float(first_token_time_match.group(1)))
                line_numbers[idx].append(i)  # 使用行号作为时间戳

# 绘制 'res_time' 折线图
plt.figure(figsize=(10, 5))
plt.plot(line_numbers[0], res_time_lists[0], label='res_time_logfile1', color='blue')
plt.plot(line_numbers[1], res_time_lists[1], label='res_time_logfile2', color='orange')
plt.xlabel('Line Number')
plt.ylabel('res_time')
plt.title('res_time over log lines')
plt.legend()
plt.grid(True)
plt.show()

# 绘制 'first_token_time' 折线图
plt.figure(figsize=(10, 5))
plt.plot(line_numbers[0], first_token_time_lists[0], label='first_token_time_logfile1', color='blue')
plt.plot(line_numbers[1], first_token_time_lists[1], label='first_token_time_logfile2', color='orange')
plt.xlabel('Line Number')
plt.ylabel('first_token_time')
plt.title('first_token_time over log lines')
plt.legend()
plt.grid(True)
plt.show()

```
下面是结果对比：

![iamge_faster1]()

![iamge_faster1]()

可以看出最新的优化方法在两方面的耗时都大大改善。

## 超参配置介绍

Decoding阶段的batchsize调整为128，增加在decoding阶段的并发推理数量。

```
model_config:
    model_name: 'llama_7b'
    max_generate_length: 4096
    end_token: 2
    seq_length: [4096]
    vocab_size: 32000
    prefill_batch_size: [1]
    decode_batch_size: [128]
    zactivate_len: [4096]
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

python test_serving_performance.py -X 5 -P 8835 -O "./" -T 300
每秒5包一共发300秒，共计1500条测试数据。


## 运行环境说明
无额外配置。

## 提交推理的日志、配置文件
https://wzybisai.obs.cn-southwest-2.myhuaweicloud.com/faster-730.zip



