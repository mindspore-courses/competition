# 推理算法介绍（llm-serving + performance-serving）

## 摘要
• 本次任务中的优化通过llm-serving的超参配置，寻找到npu-ai core最大使用情况下，能够并行处理的最优参数。

• llm-serving的推理流程，我是通过哪些log和文件来分析工程的。或许工程师之后可以写一个README。

• 以及我调参的依据和未来可做的方向。

为了简明，本文不黏贴过多代码，主要以伪代码形式讲解逻辑。


## 本次题目要求和领域分析

分析本次题目，主要调用了 llm-serving来作为 服务器端的推理框架，底层基于mindformer和mindspore来实现模型的加载和推理。使用performance_serving来作为客户端，输入问题的json文
件，在指定时间间隔发送request到客户端，执行推理任务。

以 本次 题目的baseline为例，进行分析，从而优化方向和要调研的论文。

```
python test_serving_performance.py -X 0.5 -P 8835 -O "./" -T 3000
```

理论上，按照这样的发送速率，最快能够达到0.5*3000s = 1500内完成推理。但这要求，在发送的时间间隔内(2s)就能完成任务，否则就会导致请求停留在服务器的队列中，超过2s就会拖延最终处理时间。baseline中3500s的结果就证明，单次推理超过了2s的时间间隔。而在llm-serving的现有框架下，如果想要加快推理速度，只有启用ge，但是我看到模型的ge推理处于注释状态，我猜测可能是废弃状态。因此在单个模型推理加速方面，要优化，可能需要较为苦难的处理，这个可以留作复赛彩蛋，但是目
前不去尝试。

但是这样的速度，是按照串行执行的速率，也就是说，每次只推理一个请求。而我们知道，深度学习的推理，往往能够将多个输入的句子组成batching，输入给device，进行并行的推理。而观测到llmserving实际上已经实现了continuous batching，不仅能够并行，还能够实现动态的剔除完成的句子
并加入新句子。而之所以baseline只有3500，是因为prefill_batch_size: [1] 和
decode_batch_size: [1] 的设置都为1，只需要将其调大，就能够实现加速。

其次是访存的优化，由于llm在推理的过程中，每次只需要用最新的一个Q和K，V来输入，然后将KV和
陈旧的KV-cache拼在一起进行运算就能得到等价的结果。因此在推理长句子的时候需要申请大量的内
存。观测到llm-serving存在PagedAttention的实现，我们只需要启用就可以了。

## llm-serving架构分析

文件分析

- 启动是通过examples的start.py作为入口

- 超参配置在 llm-serving/configs/llama/llama_7b_kbk_pa_dyn.yaml中完成

- 主要代码都在 llm-serving/mindspore_serving中

  - subprocess启动子进程
  
  - server用于服务器收发
  
- 其中agent中的 agent_multi_post_method.py包含主要的业务代码。


### 服务器数据接收

• 如示意伪代码所示，数据由performance-serving发送到llm-serving接收，接收后，传递给
WorkAgent，应用其中的llama-7b模型进行推理

```
performance_serving : client >> data()
|
conn.sendall() <<>> conn.recv(4096)
|
llm-serving : data >> server(socket) >> WorkerAgent(Shared_memory) >> predict
```

## llm-serving推理流程（agent_multi_post_method.py）

start.py > start_agent.py > agent_multi_post_method.py 其中通过子进程来进
行调用。

这里详解agent_multi_post_method 的推理逻辑，首先调用到 startup_agents ，开启子进
程 target=start_agent_socket_server

## start_agent_socket_server 数据收发和推理 解析

```
#伪代码
#初始化一个workagent
work_agent = WorkAgent(i, cfg)
#创建一个server，并listen客户端
agent_address = (agent_ip, agent_ports[i])
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(agent_address)
server.listen(5)
#while True 循环，不断读取数据进行处理
while True:
  data = conn.recv(4096)
  elif data.startswith("*"):
  #全量推理
  _, _ = work_agent.predict(shape_list=input_shapes)
  elif data.startswith("a"):
  #增量推理
  #这里是调用workAgent的predict
```

## WorkAgent 推理数据流 解析

前处理 >> 推理 >> 后处理

- Predict: 主要是 前处理（组batching和申请内存）和 后处理，中间调用 predict for kbk来做模型
推理

```
一大段代码

```


- decode_index和input_ids都是什么？


## 搞懂关键变量：从而搞懂 PA 和 CB

```
#predict的三个输入
shape_list
[0]:first_group >> current_index
[1]: gen_params
[2]:block_tables_shape
[3]:slot_mapping_shape


current_batch 表示 当前batch的数量（主要用于 continuous batching）
batch_valid_flag

```
- Predict_for_kbk:

```
if self.mindspore_model.config.use_past:
    logging.debug(f"predict before pa predict_for_kbk.")
    if self.is_prefill:
        self.mindspore_model.is_first_iteration = True
    #这是一个前向推理
    res, current_index = self.mindspore_model.forward(input_ids=input_ids,valid_length_each_example=valid_length,

generation_config=self.mindspore_model.config,
                                          block_tables=block_tables,
                                          slot_mapping=slot_mapping,
                                          prefill=self.is_prefill,
                                          **model_kwargs)

else:
    res = self.mindspore_model(**model_inputs)

```

## performance-serving的使用分析
业务代码主要在 performance_serving/test_serving_performance.py 中

### 入口Test_main

- 创建请求线程
    - 创建testcases,制作testcase放入列表。
    - 计算每次发送的间隔时间。
    - 创建thread_tasks，并根据testcases创建线程，存在列表当中
- 发送请求线程
    - 在列表中按生成的时间间隔迭代thread执行 task.start()，执行run函数。
- run函数会执行 send_request
- send_request 会进行
    - 所有的thread执行 task.join()，阻塞。


## 调参和依据
原始是串行执行的速率，也就是说，每次只推理一个请求。

• llm-serving实际上已经实现了continuous batching，将多个输入的句子组成batching，输入给
device，进行并行的推理。不仅能够并行，还能够实现动态的剔除完成的句子并加入新句子。而之
所以baseline只有3500，是因为prefill_batch_size: [1] 和
decode_batch_size: [1] 的设置都为1，只需要将其调大，极限为5，就能够实现加速。

• 访存的优化，由于llm在推理的过程中，每次只需要用最新的一个Q和K，V来输入，然后将KV和陈
旧的KV-cache拼在一起进行运算就能得到等价的结果。因此在推理长句子的时候需要申请大量的内
存。观测到llm-serving存在PagedAttention的实现，我们可以调整PagedAttention的参数
num_blocks: 2048 、block_size: 16 来实现优化。


## 超参配置
测速读超参配置
llm-serving/configs/llama/llama_7b_kbk_pa_dyn.yaml
启用了continous batching，经过测试，可以设成，占用npu的ai core巅峰到达83%，

## 未来可优化方向分析

• 微软DeepSpeed团队2023年11月在MII项目中提出了一种Continous Batching变种
SplitFuse.SplitFuse的想法是，对长prompt request被分解成更小的块，并在多个forward step中
进行调度，只有最后一块的forward完成后才开始这个prompt request的生成。对短prompt
request将被组合以精确填充step的空隙。每个step的计算量基本相等，达到所有请求平均延迟更
稳定的目的

• 商汤发布的pythonic LLM serving框架，简单高效，易于二次开发，和其他框架的集成。和vLLM不
同，它的prefill和decoding可以在一个step中打包成一个Batch处理.它改进了PagedAttention，弄
成tokenAttn，也就是pagedattn的page size=1，也支持了FastGen的SpliteFuse方法。














