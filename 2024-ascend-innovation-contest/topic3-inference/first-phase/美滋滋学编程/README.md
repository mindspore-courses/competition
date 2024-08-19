# 美滋滋学编程-推理调优作品报告

## 业界推理优化算法调研

#### flash attention

FlashAttention旨在**加速**注意力计算并**减少内存占用**。FlashAttention利用底层硬件的内存层次知识，例如GPU的内存层次结构，来提高计算速度和减少内存访问开销。 FlashAttention的核心原理是通过将输入**分块**并在每个块上执行注意力操作，从而减少对高带宽内存（HBM）的读写操作。具体而言，FlashAttention使用平铺和重计算等经典技术，将输入块从HBM加载到SRAM（快速缓存），在SRAM上执行注意力操作，并将结果更新回HBM。FlashAttention减少了内存读写量，从而实现了**2-4倍**的时钟时间加速。

#### paged attention

PagedAttention 的提出是为了解决大模型推理中 KV Cache 带来的显存空间利用率低的问题，该问题的主要原因在于现有的推理系统将 KV Cache 存储在连续的显存空间中，导致：

1. 内部碎片和外部碎片：由于 KV Cache 占用的显存大小随着 seq_len 动态变化，而对于不同的请求输入我们无法预先确定模型的输出序列长度，所以对于每个请求都需要预留 max_seq_len 对应的显存大小给 KV Cache。而在推理过程中所需要的 KV Cache 大小可能比预留的大小要小得多，但预留的这部分显存在请求的整个生命周期都被保留，未被使用的部分无法被其他请求利用，导致内部碎片严重。另一方面，外部内存碎片也可能很严重，因为每个请求的 max_seq_len 可能不同。
2. 无法进行内存共享：LLM 服务通常使用先进的解码算法，例如 parallel sampling 和 beam search，这些解码算法会为每个请求产生多个输出。在这些场景中，单个请求由多个序列（sequence）组成，这些序列具有公共前缀，它们可以共享 kv cache。然而，在现有系统中，内存共享是不可能的，因为每个序列的 kv cache 存储在单独的连续空间中，无法被拆出一部分进行共享。

受到操作系统使用带分页的虚拟内存解决了内存碎片和共享的方案启发，PagedAttention 将请求的 KV Cache 划分成固定大小的块（blocks），每个 block 存储固定数量 tokens 对应的 KV Cache 数据。在 PagedAttention 中，KV Cache 的 blocks 不一定存储在连续的空间中。因此，我们可以像操作系统的虚拟内存一样以更灵活的方式管理 KV Cache：将 block 看作页，将 token 看作字节，将 sequence 看作进程。这种设计通过使用相对较小的块并按需分配它们来减轻内部碎片。此外，它消除了外部碎片，因为所有块都具有相同的大小。最后，它支持以 block 为粒度，跨同一请求关联的不同序列甚至跨不同请求的内存共享。

#### continuous batching

由于 LLM 巨大的 GPU 内存开销和计算成本，在大多数应用中，机器学习工程师通常通过内部调整（如量化和对 CUDA 核的定制）来优化。然而，由于 LLM 通过迭代生成其输出，并且 LLM 推理通常涉及内存而不是计算，因此在很多实践中，优化系统级批处理可以使性能差异达到10倍甚至更多。

一种最近提出的优化方法是连续批处理（Continuous batching），也称为动态批处理或基于迭代级的批处理。其具有如下惊人的效果：

基于vLLM，使用连续批处理和连续批处理特定的内存优化，可以实现多达23倍的吞吐量提升；

对于 HuggingFace 本地生成推理，使用连续批处理，可以实现8倍的吞吐量提升；

基于 NVIDIA 的 FasterTransformer，使用优化过的模型实现，可以实现4倍的吞吐量提升。

## 本作品使用的推理优化算法介绍

==启动flash attention==，增加推理速度。

==增大decoding部分的batchsize数==，结果多轮测试，128batchsize情况下推理系统吞吐量最优。

==将mindspore的context启动图算融合==。

==调整句子输出长度上限==。 该方法虽然输出文本完全一致，但是不满足精度测试要求，遂没有采用。



对于server.log日志进行分析，分析提交给系统的推理请求，查找最大输出长度。

```python
import re

def find_max_token_len(log_file_path):
    # 定义正则表达式来匹配 'max_token_len': 数字
    pattern = re.compile(r"'max_token_len': (\d+)}")
    max_value = 0

    try:
        with open(log_file_path, 'r') as file:
            for line in file:
                match = pattern.search(line)
                if match:
                    value = int(match.group(1))
                    if value > max_value:
                        max_value = value
    except FileNotFoundError:
        print(f"文件 {log_file_path} 未找到.")
        return
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return

    print(f"最大的 'max_token_len' 数值是: {max_value}")

# 使用示例
log_file_path = 'server.log'
find_max_token_len(log_file_path)
```



测速数据集的最大输出长度为504，精度数据集最大的输出长度为464.

由于包含原始输入，所以如果将句子长度设置为512，整体的输出长度会超过该限制。

对应的错误输出内容如下：

```
test_llama.log - INFO - {'input': 'Develop a survey to collect customer feedback', 'resp_text': 'Error202: prompt out of range', 'res_time': 0.7274570465087891, 'first_token_time': 0.6255989074707031}
```

将句子长度调整为520刚刚好，但这种方法会造成最终词表概率值发生细微变化，不满足精度测试要求。所以虽然可以提升推理服务吞吐量，但最终的测速没用上该方法。



==优先级队列调整prefill和decoding任务调度==。

描述：将decoding的batch增大以后，发现即使是batch推理，其中还是有很多的padding位置，造成计算资源的大量浪费。所以就有一个想法是将接收到的data放入到优先级任务队列中，将decoding任务作为低优先级，其他的任务作为高优先级任务。优先将prefill阶段的任务执行完毕后，后面组成的decoding阶段的batch就会真正打满，从而加速推理运行。

核心代码：

```python
def start_agent_socket_server(i, cfg: ServingConfig, startup_queue):
    logging.basicConfig(level=logging.ERROR,
                        filename=f"./output/agent_{i}.log",
                        filemode='w',
                        format=
                        '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    """启动agent进程, 由_agent_process进行调用, 创建agent进程"""
    if IMPORT_LITE_FAILED:
        logging.warning("import mindspore_lite failed, using kbk backend.")
    work_agent = WorkAgent(i, cfg)  # 创建一个WorkAgent实例，传入当前agent的索引和配置。

    agent_ports = cfg.serving_config.agent_ports
    agent_ip = cfg.serving_config.agent_ip
    agent_address = (agent_ip, agent_ports[i])
    # 设置当前agent的地址（IP和端口）。
    print(agent_address)

    parent_process = psutil.Process(os.getppid())
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(agent_address)
    server.listen(50)  # 开始监听连接，允许最多50个待处理连接

    startup_queue.put(i)

    # 绑定method
    print("start agent socket server in rank{}".format(i), flush=True)
    logging.info("Agent socket server started on {}".format(agent_address))

    task_queue = queue.PriorityQueue()

    def handle_client(conn):
        while True:
            if not parent_process.is_running():
                logging.warning(
                    f"detect parent pid={parent_process.pid} has exited, child begin to exit")
                conn.close()
                return

            try:
                data = conn.recv(4096)
                if not data:
                    break
                data = data.decode()
                logging.debug(f"Data received: {data}")

                if data.startswith('#') or data.startswith('*') or data.startswith('e') or data.startswith('r'):
                    priority = 0  # 高优先级
                else:
                    priority = 1  # 低优先级

                task_queue.put((priority, data, conn))
                logging.info(f"Task added to queue with priority {priority}: {data}")

            except ConnectionResetError:
                break
            except RuntimeError as e:
                logging.error(f"Runtime error: {e}")
                conn.sendall("2".encode())
                break

    def process_tasks():
        while True:
            priority, data, conn = task_queue.get()
            logging.info(f"Processing task with priority {priority}: {data}")

            if data.startswith('#'):
                if work_agent.status & AgentStatus.unconnected == AgentStatus.unconnected:
                    data = data[1:]
                    work_agent.shm_names = data.split(",")
                    work_agent.status = AgentStatus.connected
                    logging.info("Connected successfully")
                    conn.sendall("success".encode())
                else:
                    logging.info("Connection failed")
                    conn.sendall("failed".encode())

            elif data.startswith('*'):
                work_agent.is_prefill = True
                data = data[1:]
                shape_strs = data.split(",")
                input_shapes = [list(map(int, shape_str.split(" "))) for shape_str in shape_strs]
                work_agent.predict(shape_list=input_shapes)
                if i == 0:
                    conn.sendall("1".encode())

            elif data.startswith('a'):
                decode_data = data.split('_')
                current_batch_dyn = int(decode_data[-4]) if cfg.model_config.page_attention else int(decode_data[-2])
                batch_valid_flag = [int(ele) for ele in (decode_data[-3] if cfg.model_config.page_attention else decode_data[-1]).split(" ")]
                input_shapes = []
                if cfg.model_config.page_attention:
                    input_shapes = [list(map(int, decode_data[idx].split(" "))) for idx in [-2, -1]]
                work_agent.is_prefill = False
                work_agent.predict(current_batch=current_batch_dyn, batch_valid_flag=batch_valid_flag, shape_list=input_shapes)
                if i == 0:
                    conn.sendall("1".encode())

            elif data.startswith('e'):
                if work_agent.status & AgentStatus.busy == AgentStatus.busy:
                    logging.info("Agent is busy")
                    conn.sendall("busy".encode())
                else:
                    work_agent.status = AgentStatus.unconnected
                    logging.info("Agent is free")
                    conn.sendall("free".encode())

            elif data.startswith('r'):
                work_agent.status = AgentStatus.unconnected
                logging.info("Reset successful")
                conn.sendall("success".encode())

    threading.Thread(target=process_tasks, daemon=True).start()

    while True:
        if not parent_process.is_running():
            logging.warning(f"detect parent pid={parent_process.pid} has exited, child begin to exit")
            server.close()
            return
        conn, client_addr = server.accept()
        logging.info(f"Connection accepted from {client_addr}")
        threading.Thread(target=handle_client, args=(conn,), daemon=True).start()
```



但是目前这种方法貌似有点bug，推理的输出长度没有问题，但是输出的内容比较错乱。比赛时间紧张，只能先提交一版有bug的版本上来，希望后续有机会继续完善。

使用该优化方法推理时间如下，但是由于存在bug，所以目前提交的最终版本是没有使用这种策略的，但是从推理耗时来看确实有很大提升。

```
248.1099133491516 //test_performance_2024-07-29-21_33.log
```




## 超参配置介绍

测速脚本配置：

```
python test_serving_performance.py -X 10 -P 8835 -O "./" -T 150
```

第一条命令用于预热NPU，确保后续推理请求处理比较稳定。

每秒发送10包，一共发送150秒。共计1500条推理请求。



llm-serving/configs/llama/llama_7b_kbk_pa_dyn.yaml

```yaml
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
    model_dtype: "DataType.FLOAT16"
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



mindformers/configs/llama2/predict_llama2_7b.yaml

```yaml
# mindspore context init config
context:
  mode: 0 #0--Graph Mode; 1--Pynative Mode
  device_target: "Ascend"
  enable_graph_kernel: True #开启图算融合
  graph_kernel_flags: "--disable_expand_ops=Softmax,Dropout --enable_parallel_fusion=true --reduce_fuse_depth=8 --enable_auto_tensor_inplace=true"
  max_call_depth: 10000
  max_device_memory: "58GB" #增大内存占用
  save_graphs: False
  save_graphs_path: "./graph"
  device_id: 0
  
  use_flash_attention: True # FA can accelerate training or finetune
```



### 运行环境说明

环境均按照1.6.2环境配置进行，没有额外的环境配置。

### 提交推理的日志、配置文件

附件-test_performance_2024-07-31-00_32.log

### 提交llm-serving源码包和performance源码包

https://tuilizhuany.obs.cn-southwest-2.myhuaweicloud.com/731.zip
