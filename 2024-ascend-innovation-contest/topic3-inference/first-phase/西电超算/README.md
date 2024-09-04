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
# 最重要的类方法是 predict
def predict(self, shape_list=None, current_batch=None, batch_valid_flag=None):
    self.status = AgentStatus.busy
    #申请tmp_shms，包括 existing_shm0，output_shm，output_logprob_shm，gen_params_shm
    tmp_shms = []
    start_time = time.time()
    existing_shm0 = shared_memory.SharedMemory(name=self.shm_names[0])  #共享内存
    tmp_shms.append(existing_shm0)

    # prefill阶段，只进行各种预处理，没有调用模型

    #########################################################
    if self.is_prefill:
        # 第一个数据块的形状
        first_group = np.ndarray((shape_list[0]), dtype=np.int32, buffer=existing_shm0.buf)
        # 当前的index，是从first_group中进行切片得到
        current_index_ = first_group[:, shape_list[0][1] - 3: shape_list[0][1] - 2]
        current_index = np.squeeze(current_index_, axis=-1) # 去除冗余的长度为1的轴

        valid_length_ = first_group[:, shape_list[0][1] - 1: shape_list[0][1]]
        if self.config.model_config.current_index or self.config.model_config.backend == "kbk":
            valid_length = np.squeeze(valid_length_, axis=-1).astype(np.int64)
        else:
            valid_length = np.squeeze(valid_length_, axis=-1).astype(np.int32)

        input_ids = first_group[:, :shape_list[0][1] - 3
        # 生成参数 由id 指定在 shape list中的位置，来进行生成
        gen_params_id = 1  # 改为1，正向取值，原始shape_list只有两个值，现在多加了两个
        shape_params = shape_list[gen_params_id]
        gen_params = np.ndarray(shape_params, dtype=np.float16, buffer=gen_params_shm.buf)
        # 生成参数 包括 do_sample_list、top_p_list、top_k_list、temperature_list、repetition_penalty_list、decode_index_list

        do_sample_list = gen_params[:, 0].astype(np.bool_)
        top_p_list = gen_params[:, 1]
        top_k_list = gen_params[:, 2].astype(np.int32)
        temperature_list = gen_params[:, 3]
        repetition_penalty_list = gen_params[:, 4]
        decode_index_list = gen_params[:, 5].astype(np.int32)
        # 添加baichuanPA block_tables_shape slot_mapping_shape
        if self.config.model_config.page_attention:
            block_tables_shape = shape_list[2] # block_tables_shape 存储了块表格的形状信息
            slot_mapping_shape = shape_list[3] # 如何将输入数据的不同部分映射到模型的处理单元或输出单元

        extra_input = []
        # 创建新的 existing_shm用于存放 extra_input
        for i in range(1, len(shape_list) - 1): #除了第一个是真实的 input，后面是参数
            existing_shm = shared_memory.SharedMemory(name=self.shm_names[i])
            tmp_shms.append(existing_shm)
            # To Do np.int64 ?
            extra_input.append(np.ndarray((shape_list[i]), dtype=np.int64, buffer=existing_shm.buf))

        if self.config.model_config.backend == "ge":
            # pa or static model type don't need 'act_len' parameter
            if self.config.model_config.page_attention or (
                    self.config.model_config.model_name == 'wizard_coder' and self.config.model_config.model_type == "static"):
                extra_input = []
            else:
                extra_input = self.extra_input_func.get_extra_inputs(input_ids, current_index, None, True,
                                                                     valid_length,
                                                                     zactivate_len=self.config.model_config.zactivate_len)
        # 意味着 batch_size 已经在 input_ids中了，而 input_ids来源于 first_group[:,: shape_list[0][1] - 3]
        self.current_batch_size = len(input_ids)
        # init和decode的东西
        init_reset = []  # 存储初始化值
        decode_index = []  # 存储 decode的下标
        for i in range(self.current_batch_size):
            decode_params = DecodeParams(
                do_sample=bool(do_sample_list[i]),
                top_p=top_p_list[i],
                top_k=int(top_k_list[i]),
                temperature=temperature_list[i],
                repetition_penalty=repetition_penalty_list[i],
                decode_index=int(decode_index_list[i]),
                current_index=int(current_index[i]),
                valid_length=int(valid_length[i]),
                init_reset=False
            )
            self.decode_params_map[decode_params.decode_index] = decode_params
            init_reset.append(decode_params.init_reset)
            decode_index.append(decode_params.decode_index)
        init_reset = np.array(init_reset, dtype=np.bool_)
        decode_index_np = np.array(decode_index, dtype=np.int64)
    # decode阶段 实际进行推理
    else:
        # keep decode map size equal to current batch size
        # extend
        # 创建化列表和变量
        current_index = []
        valid_length = []
        init_reset = []
        decode_index = []
        self.current_batch_size = current_batch  # 成员变量等于current_batch
        current_batch_size = self.current_batch_size  # 局部变量等于成员变量
        # size 和 valid flag数量应该一致
        if self.current_batch_size != len(batch_valid_flag):
            batch_valid_flag.clear()
            batch_valid_flag = [1 for _ in range(self.current_batch_size)]
        
        # keys代表了before的bc
        before_batch_size = len(self.decode_params_map.keys())
        # 如果之前的少于现在的，那说明添加了新的进来了（CB），需要进行padding
        if before_batch_size < current_batch_size:
            #通过output_shm创建input_ids
            input_ids = np.ndarray((before_batch_size,), dtype=np.int32, buffer=output_shm.buf)
            # pad的符号是 2
            pad_input_id = self.config.model_config.end_token
            #添加的长度
            add_length = self.current_batch_size - before_batch_size
            addition_input_ids = np.array(add_length * [pad_input_id], dtype=np.int32)  # 额外增加的ids
            input_ids = np.append(input_ids, addition_input_ids)
            target_batch = self.current_batch_size  # 当前的 batch就是 target batch
            pad_key = list(self.decode_params_map.keys())[-1]
            # padding_obj = self.decode_params_map[pad_key]
            for j in range(target_batch):
                if j not in self.decode_params_map:
                    padding_obj = copy.deepcopy(self.decode_params_map[pad_key])
                    padding_obj.current_index = 0
                    padding_obj.valid_length = 1
                    padding_obj.decode_index = j
                    self.decode_params_map[j] = padding_obj
        else:
            # pop
            while len(self.decode_params_map.keys()) > current_batch_size:
                #从 decode_params_map 中弹出，self.decode_params_map[decode_params.decode_index] = decode_params
                self.decode_params_map.popitem()
            input_ids = np.ndarray((current_batch_size,), dtype=np.int32, buffer=output_shm.buf)

        # 给键值 排序
        self.decode_params_map = dict(sorted(self.decode_params_map.items(), key=lambda x: x[0]))
        # 很朴素的 全部 +1
        for key in self.decode_params_map.keys():
            decode_params = self.decode_params_map[key]
            decode_params.current_index = decode_params.current_index + 1
            decode_params.valid_length = decode_params.valid_length + 1
            decode_params.init_reset = True  # 修改原始代码bug
            if batch_valid_flag[key] == 1:
                current_index.append(decode_params.current_index)
                valid_length.append(decode_params.valid_length)
            else:
                current_index.append(0)
                valid_length.append(1)
            init_reset.append(decode_params.init_reset)
            decode_index.append(decode_params.decode_index)

        if self.config.model_config.backend == "ge":
            # pa or static model type don't need 'act_len' parameter
            if self.config.model_config.page_attention or (
                    self.config.model_config.model_name == 'wizard_coder' and self.config.model_config.model_type == "static"):
                extra_input = []
            else:
                extra_input = self.extra_input_func.get_extra_inputs(input_ids, current_index, None, False,
                                                                     valid_length,
                                                                     zactivate_len=self.config.model_config.zactivate_len)
        #重新创建 np.array
        current_index = np.array(current_index, dtype=np.int32)
        if self.config.model_config.current_index or self.config.model_config.backend == "kbk":
            valid_length = np.array(valid_length, dtype=np.int64)
        else:
            valid_length = np.array(valid_length, dtype=np.int32)
        # 重整三个 关键的矩阵
        init_reset = np.array(init_reset, dtype=np.bool_)
        decode_index_np = np.array(decode_index, dtype=np.int64)
        input_ids = input_ids.reshape((-1, 1))
        # 加入PA特性
        #PA:这个时候 再次 确定 块表的尺寸和 插槽映射的尺寸 #############
        if self.config.model_config.page_attention:
            block_tables_shape = shape_list[0]
            slot_mapping_shape = shape_list[1]

    #这里开始不分 prefill和decode

    block_tables_np = None
    slot_mapping_np = None
    if self.config.model_config.page_attention:  # 当然是要用 page_attention了
        # 创建共享内存
        block_tables_shm = shared_memory.SharedMemory(name=self.shm_names[7])  # 这里的共享内存index要改
        slot_mapping_shm = shared_memory.SharedMemory(name=self.shm_names[8])
        # 根据shape，在指定的内存，创建array
        block_tables_np = np.ndarray((block_tables_shape), dtype=np.int32, buffer=block_tables_shm.buf)
        slot_mapping_np = np.ndarray((slot_mapping_shape), dtype=np.int32, buffer=slot_mapping_shm.buf)


    if self.config.model_config.backend == "ge":
        if self.config.model_config.page_attention:
            if self.is_prefill:
                tmp_in = [input_ids, valid_length, slot_mapping_np]
            else:
                tmp_in = [input_ids, valid_length, block_tables_np, slot_mapping_np]
        else:
            tmp_in = self.basic_input_func.get_inputs(input_ids, current_index, init_reset, valid_length,
                                                      self.config.model_config.current_index, decode_index_np,
                                                      self.config.model_config.model_type)
            if len(extra_input) > 0:
                tmp_in.extend(extra_input)

        for tmp in tmp_in:
            print(1)

        outputs = self.predict_for_ge(extra_input, start_time, tmp_in)
    else:
        # 得到 seq_length,目前生成了多长
        seq_length = self._get_seq_length(input_ids, False)
        # init kbk_targets, shape(current_batch, seq_length), default value: self.config.model_config.pad_token_id
        # 如果 kbk_targets还不存在，就创建一个
        if self.kbk_targets is None:
            decode_batch_size = self.config.model_config.decode_batch_size[0]
            # bs * len 用pad 2 来填充
            self.kbk_targets = np.full((decode_batch_size, seq_length), self.config.model_config.pad_token_id)



        # decode 时，先将 shape 与 prefill 改为一致
        # 如果 input_ids.shape[1]是 1，意味着 是prefill？不确定
        if input_ids.shape[1] == 1:
            # 如果初始时 input_ids 的第二个维度为1，那么将其扩展到指定的 seq_length
            input_ids = np.concatenate((input_ids, np.zeros((input_ids.shape[0], seq_length - 1))), axis=1)

        # 遍历 decode_index，decode_index 通常指的是用于指导生成过程中位置或者词汇选择的索引

        # 遍历decode_index
        for idx, index in enumerate(decode_index):
            index = int(decode_index[0])

            if self.is_prefill:
                self.kbk_targets[index] = input_ids[idx]
            else:
                current_index_value = int(current_index[idx])
                # 精彩的复制，就是把input_ids切片给kbk_targets了
                self.kbk_targets[index][current_index_value:current_index_value + 1] = input_ids[idx][:1]
                input_ids[idx] = self.kbk_targets[index]

        # 上面都是预处理，这里是进行一次前向传播

        outputs = self.predict_for_kbk(current_index, input_ids, valid_length, block_tables_np, slot_mapping_np)
    # 后处理
    post_time = time.time()
    if self.rank_id == 0:
        multi_thread_time = time.time()
        
        if self.is_prefill:
            self.do_post_sampling(outputs, output_shm, output_logprob_shm, decode_index_np, prefill=True)
        else:
            self.do_post_sampling(outputs, output_shm, output_logprob_shm, decode_index_np, prefill=False)

    self.status &= ~AgentStatus.busy
    return self.targets, tmp_shms

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
启用了continous batching，经过测试，可以设成，占用npu的ai core巅峰到达83%。

```
model_config:
    model_name: 'llama_7b'
    max_generate_length: 4096
    end_token: 2
    seq_length: [4096]
    vocab_size: 32000
    prefill_batch_size: [96]
    decode_batch_size: [96]
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

测速指令(performance_serving)：

```
python test_serving_performance.py -X 3 -P 1 8835 -O "./" -T 500
```

精度测试配置：

由于 logits的保存脚本，不兼容continuous batching，因此没有办法并行的保存logits。

因此精度的测试，其实就是按照baseline来做，需要将 batch 调整回1。我这里进行了验证，得到了正
确的结果。


## 未来可优化方向分析

• 微软DeepSpeed团队2023年11月在MII项目中提出了一种Continous Batching变种
SplitFuse.SplitFuse的想法是，对长prompt request被分解成更小的块，并在多个forward step中
进行调度，只有最后一块的forward完成后才开始这个prompt request的生成。对短prompt
request将被组合以精确填充step的空隙。每个step的计算量基本相等，达到所有请求平均延迟更
稳定的目的

• 商汤发布的pythonic LLM serving框架，简单高效，易于二次开发，和其他框架的集成。和vLLM不
同，它的prefill和decoding可以在一个step中打包成一个Batch处理.它改进了PagedAttention，弄
成tokenAttn，也就是pagedattn的page size=1，也支持了FastGen的SpliteFuse方法。














