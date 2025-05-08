

# **作品介绍：**

团队名：起风了

## 一、优化策略

### 1. 超参数调优

调整decode_batch_size: [128]，修改llama_7b_kbk_pa_dyn.yaml文件,时间来到680s左右

```
model_config:
    model_name: 'llama_7b'
    # max_generate_length: 600 ##快几秒
    max_generate_length: 4096
    end_token: 2
    seq_length: [4096]
    vocab_size: 32000
    prefill_batch_size: [1]
    # decode_batch_size: [1]
    decode_batch_size: [128]
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



### 2. 调整prefill和decoding任务调度

发现初始的版本decode阶段还是存在大量padding, 没有利用好显卡的并行计算优势，AIcore利用率大致为60%左右

为了增加decode的并行度，减少无效padding, 将prefill请求优先级调整到最高，decoding任务优先级最低，这样可以使得decode阶段的并行请求更多，尽可能打满整个batch，AIcore利用率提升至为80%左右，时间来到610s左右

具体修改如下：

1）修改agent_multi_post_method.py文件

```
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
    # print("start agent socket server in rank{}".format(i), flush=True)
    # logging.info("Agent socket server started on {}".format(agent_address))

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
                # logging.debug(f"Data received: {data}")

                if data.startswith('#') or data.startswith('*') or data.startswith('e') or data.startswith('r'):
                    priority = 0  # 高优先级
                else:
                    priority = 1  # 低优先级

                task_queue.put((priority, data, conn))
                # logging.info(f"Task added to queue with priority {priority}: {data}")

            except ConnectionResetError:
                break
            except RuntimeError as e:
                logging.error(f"Runtime error: {e}")
                conn.sendall("2".encode())
                break

    def process_tasks():
        while True:
            priority, data, conn = task_queue.get()
            # logging.info(f"Processing task with priority {priority}: {data}")

            if data.startswith('#'):
                if work_agent.status & AgentStatus.unconnected == AgentStatus.unconnected:
                    data = data[1:]
                    work_agent.shm_names = data.split(",")
                    work_agent.status = AgentStatus.connected
                    # logging.info("Connected successfully")
                    conn.sendall("success".encode())
                else:
                    # logging.info("Connection failed")
                    conn.sendall("failed".encode())
            
            elif data.startswith('*'):
                    # 全量推理
                    work_agent.is_prefill = True
                    data = data[1:]
                    shape_strs = data.split(",")
                    input_shapes = []
                    for shape_str in shape_strs:
                        shape = list(map(int, shape_str.split(" ")))
                        input_shapes.append(shape)
                    _, _ = work_agent.predict(shape_list=input_shapes)
                    if i == 0:
                        conn.sendall("1".encode())
                        
            elif data.startswith('a'):
                    # 增量推理
                    decode_data = data.split('_')
                    # 增加PA的判断
                    current_batch_dyn = int(decode_data[-4]) if cfg.model_config.page_attention else int(
                        decode_data[-2])
                    batch_valid_flag = []
                    batch_valid = decode_data[-3] if cfg.model_config.page_attention else decode_data[-1]
                    for ele in batch_valid.split(" "):
                        batch_valid_flag.append(int(ele))
                    # 增加 block_tables和slot_mapping 的 shape
                    input_shapes = []
                    if cfg.model_config.page_attention:
                        for shape_str in [decode_data[-2], decode_data[-1]]:
                            shape = list(map(int, shape_str.split(" ")))
                            input_shapes.append(shape)
                    work_agent.is_prefill = False
                    _, _ = work_agent.predict(current_batch=current_batch_dyn, batch_valid_flag=batch_valid_flag,
                                              shape_list=input_shapes)
                    if i == 0:
                        conn.sendall("1".encode())
            elif data.startswith('e'):
                if work_agent.status & AgentStatus.busy == AgentStatus.busy:
                    # logging.info("Agent is busy")
                    conn.sendall("busy".encode())
                # else:
                    work_agent.status = AgentStatus.unconnected
                    # logging.info("Agent is free")
                    conn.sendall("free".encode())

            elif data.startswith('r'):
                work_agent.status = AgentStatus.unconnected
                # logging.info("Reset successful")
                conn.sendall("success".encode())

    threading.Thread(target=process_tasks, daemon=True).start()

    while True:
        if not parent_process.is_running():
            logging.warning(f"detect parent pid={parent_process.pid} has exited, child begin to exit")
            server.close()
            return
        conn, client_addr = server.accept()
        # logging.info(f"Connection accepted from {client_addr}")
        threading.Thread(target=handle_client, args=(conn,), daemon=True).start()
        
```

2）仅依靠修改优先级会导致首token得到的是无效token，同时最后会少输出一个token, 不过总体token数是正常的，如图所示，human就是读取的无效token产生的

![image-20241031164326496](C:\Users\ly\AppData\Roaming\Typora\typora-user-images\image-20241031164326496.png)

分析原因是**共享内存读取与写入的时间不同步**（答辩时再具体解析），为此还要修改如下代码：

model_init_multimodel.py文件中， call函数需要增加两行代码（位于514,515行），即首token需要返回空list（这两行在验证精度时需要注释掉，精度是没问题的，主要是因为调整prefill和decode优先级后，精度验证时保存的文件中的顺序也不一样了，所以精度验证时要关闭优先级调整策略）：

```
def call(self, shms: List, input_ids, current_index,
             valid_length, init_reset, is_first_iteration, valid_batch_flag, extra_inputs=None,
             current_batch_size=None, **kwargs):
      
		............
		............
        for item in self.agent_stubs:
            item.sendall(shapes_str)
        recv_data = self.agent_stubs[0].recv(1, socket.MSG_WAITALL).decode()
        # if not recv_data=="1":
        #     recv_data = self.agent_stubs[0].recv(1, socket.MSG_WAITALL).decode()
        result = []
        if recv_data == "2":
            for _ in decode_index_list:
                # result.append(int(Baseconfig.end_token))
                result.append((int(-1),0))
            print("--------------------predict failed, abandon current prompt, please try again----------------")
            logging.error("predict failed, abandon current prompt, please try again")
            return result, 1
        
        ####测试精度时需要注释下面两行
        if is_first_iteration:
             return result, 1
        ############

        for decode_index in decode_index_list:
            tmp = np.ndarray((decode_index + 1,), dtype=np.int32, buffer=shms[5].buf)
            tmp_logprob = np.ndarray((decode_index + 1,), dtype=np.float64, buffer=shms[6].buf)
            result.append((int(tmp[decode_index:decode_index + 1]), float(tmp_logprob[decode_index:decode_index + 1])))
        
        logging.info("--------------------callV3 result value is {} ".format(result))
        logging.info("model.call time is {} ".format((time.time() - time_start) * 1000))
        return result, 1

```

3）同时master.py需要进行如下调整，修改_postprocess（）函数

```
    def _postprocess(self,
                     outputs: List[tuple],
                     entry_metadata_list: List[EntryMetaData],
                     index_list: List[int] = None,
                     skip_inference=False) -> List[ResponseOutput]:
      
        end_token = self.config.model_config.end_token  # for debug
        
        ################ 首token是无效token,更新状态后返回，不要解码 
        if len(outputs)==0 or outputs==[]:
            self.scheduler.upate_entries_after_one_step_after_prefill(end_token, index_list)
            return None
		#################
        output_tokens = []
        output_logprob = []
        ### 整个batch，一个迭代的输出， 一个迭代输出一个token
        for output_tup in outputs:
            output_tokens.append(output_tup[0])
            output_logprob.append(output_tup[1])
        .........

```

4）schedule.py文件中，创建upate_entries_after_one_step_after_prefill函数，增加如下代码：

```
def upate_entries_after_one_step_after_prefill(self, eos_id: int, index_list: List[int] = None):
        """update status after ever iteration"""
        # optimize prefill multi-batch later
        if index_list is not None:
            # idx: index_list and outputs data index, index: batch list index.
            for idx, index in enumerate(index_list):
                self.running_request_list[index].is_prompt = False
                # invalid prompt
                if self.running_request_list[index].get_entry_data().get_status() == EntryStatus.PADDING_INVAILED:
                    continue

                if self.running_request_list[index].get_entry_data().get_status() == EntryStatus.INPUT_OUTOFRANGE:
                    update_token = INPUT_OUT_OF_TOKEN[0]
                elif self.running_request_list[index].get_entry_data().get_status() == EntryStatus.EMPTY_PROMPT_TOKEN:
                    update_token = INPUT_EMPTY_TOKEN[0]
                else:
                    continue
                
                self.running_request_list[index].get_entry_data().updata_output_tokens(update_token)
                # valid prompt 区分PA处理
                if self.config.model_config.page_attention:
                    self._finished_pa_request(index, update_token, eos_id)
                else:
                    self._finished_request(index, update_token, eos_id)
```

5）修改完以上代码，输出内容正常了

![image-20241031170331451](C:\Users\ly\AppData\Roaming\Typora\typora-user-images\image-20241031170331451.png)



### 3. 后处理和模型推理异步进行

初始版本predict（推理过程）和post_process（解码过程）是串行的，事实上模型推理完不需要等待后处理完成

因此将predict和post_process解耦，变成异步进行的方式，时间来到 606s（提升4s左右）

修改master.py文件，

```
import asyncio
from mindspore_serving.master.request_resister_engine import RequestEngine
class AsyncMaster(Master):
    def __init__(
        self,
        config: ServingConfig,
        request_engine: RequestEngine
    ):
        super().__init__(config)
        self.detokenizer_que = asyncio.Queue() ############ add
        self.request_engine = request_engine ############ add
        
    ############### add
    def send_post_process(self,output,entry_metadata_list,index_list,skip_inference=False):
        self.detokenizer_que.put_nowait(
            BatchTokenIdOut(output,entry_metadata_list,index_list,skip_inference)
        )
    ###############
    .............

    async def _run_workers_async(self, current_batch_size, entry_metadata_list):
        # e_t_e_time = time.time()

        prompt_token_empty_list = self._check_prompt_token_empty(entry_metadata_list,
                                                                 self.config.model_config.pad_token_id)
        # logging.debug("prompt token empty list index_list {}".format(prompt_token_empty_list))
        if len(prompt_token_empty_list) > 0:
            ############# add
            self.send_post_process([INPUT_EMPTY_TOKEN], entry_metadata_list=entry_metadata_list,
	                                     index_list=prompt_token_empty_list,
	                                     skip_inference=True)
            # return self._postprocess([INPUT_EMPTY_TOKEN], entry_metadata_list=entry_metadata_list,
            #                          index_list=prompt_token_empty_list,
            #                          skip_inference=True)

        # check prefill out of range data
        out_of_range_index_list = self._check_prompt_out_of_range_index_list(entry_metadata_list)
        # logging.debug("out of range prompt index_list {}".format(out_of_range_index_list))
        if len(out_of_range_index_list) > 0:
            ######################### add
            self.send_post_process([INPUT_OUT_OF_TOKEN], entry_metadata_list=entry_metadata_list,
	                             index_list=out_of_range_index_list,
	                             skip_inference=True)
            # return self._postprocess([INPUT_OUT_OF_TOKEN], entry_metadata_list=entry_metadata_list,
            #                          index_list=out_of_range_index_list,
            #                          skip_inference=True)

        # filter prompt data batch list
        ## entry_metadata_list就是schedule返回的decode_batchsize个请求
        ### 这里根据 prefill_batchsize从中取出prefill_batchsize个请求
        input_entry_metadata_list, index_list = self._get_prompt_batch_list(entry_metadata_list)
        # logging.debug("_get_prompt_batch_list prompt index_list {}, input_entry_metadata_list {}"
        #               .format(index_list, input_entry_metadata_list))
        # prefill predict
        if len(input_entry_metadata_list) > 0: ### 只要running_request_list存在刚进来的prompt, 就在先要进行prefill
            # logging.debug('prefill len of input entry_metadata_list is {}'.format(len(input_entry_metadata_list)))
            # predict
            output = self.worker.predict(current_batch_size, entry_metadata_list=input_entry_metadata_list)
        else:  # decode predict
            input_entry_metadata_list = entry_metadata_list
            index_list = None
            # logging.debug('decode len of input entry_metadata_list is {}'.format(len(input_entry_metadata_list)))
            output = self.worker.predict(current_batch_size, entry_metadata_list=input_entry_metadata_list)

        # post_process_time = time.time()
        ################################# add 
        self.send_post_process(output, entry_metadata_list=entry_metadata_list, index_list=index_list)
        # result = self._postprocess(output, entry_metadata_list=entry_metadata_list, index_list=index_list)
        # logging.info('post_process_time time is {}'.format((time.time() - post_process_time) * 1000))
        # logging.info('e-to-e time is {}'.format((time.time() - e_t_e_time) * 1000))
        # return result

	................
    ########## add
    async def handle_detokenization_loop(self):
        while True:
            try:
                recv_obj = await self.detokenizer_que.get() # BatchTokenIdOut
                post_process_time = time.time()
                request_outputs = self._postprocess(recv_obj.output, entry_metadata_list=recv_obj.entry_metadata_list, index_list=recv_obj.index_list,skip_inference=recv_obj.skip_inference)
                logging.info('post_process_time time is {}'.format((time.time() - post_process_time) * 1000))
               
                # Put the outputs into the corresponding streams.
                if request_outputs is not None:
                    for request_output in request_outputs:
                        self.request_engine.process_request_output(request_output)
                self.detokenizer_que.task_done()
            except Exception as e:
                print(e)
```

同时修改llm_server_post.py文件

```
class LLMServer:
    def __init__(self, config: ServingConfig):
        self.request_engine = RequestEngine()
        self.background_loop = None
        self.master = AsyncMaster(config, self.request_engine) # liuyang
        # self.master = AsyncMaster(config)
        self.status = 0
        self.config = config

    @property
    def is_running(self) -> bool:
        return self.background_loop is not None

    async def run_loop(self):
        while self.status:
            await self.step()
            await asyncio.sleep(0)
    
    def start_background_loop(self) -> None:
        # todo
        self.status = 1
        """Start the background loop."""
        if self.is_running:
            raise RuntimeError("Background loop is already running.")
        self.background_loop = asyncio.get_event_loop().create_task(self.run_loop())
        asyncio.get_event_loop().create_task(self.master.handle_detokenization_loop()) # add 
```



### 4. 根据input长度排序推理

为了更好的模型并行化，performance_serving发送1500条请求之前，先根据input的长度对1500条请求进行排序，按照长度从短到长发送请求，这一步时间优化约2-3s，时间来到604s

修改test_serving_performance.py

```
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="test serving performance")
    parser.add_argument("-X", "--qps", help='x req/s', required=True, type=float)
    parser.add_argument("-P", "--port", help='port, default is 8000', required=True)
    parser.add_argument("-O", "--out_dir", help='dir for saving results', required=True)
    parser.add_argument("-T", "--test_time", help='test all time, default 1h', required=False, type=int, default=3600)
    args = parser.parse_args()
    with open("./alpaca_5010.json") as f:
        alpaca_data = json.loads(f.read())
    INPUTS_DATA = []
    OUTPUTS_DATA = []
    count = 0
    input_length = []
    for data in alpaca_data:
        count+=1
        if count>1500:
            break
        input_ = data["instruction"] + ":" + data["input"] if data["input"] else data["instruction"]
        INPUTS_DATA.append(input_)
        OUTPUTS_DATA.append(data["output"])
        input_length.append(len(input_))
    indexes = np.argsort(input_length)
    INPUTS_DATA = [INPUTS_DATA[i] for i in indexes]
    OUTPUTS_DATA = [OUTPUTS_DATA[i] for i in indexes]
    test_main(args.port, INPUTS_DATA, OUTPUTS_DATA, args.qps, args.out_dir, args.test_time)
```



### 5. 算子替换

将agent_multi_post_method.py中数据预处理的np.concatenate改为np.pad, 这一步提升大概6s左右，时间来到598s

```
# decode 时，先将 shape 与 prefill 改为一致
if input_ids.shape[1] == 1:
     # input_ids = np.concatenate((input_ids, np.zeros((input_ids.shape[0], seq_length - 1))), axis=1)
      input_ids = np.pad(input_ids,((0,0),(0,seq_length - 1)),'constant',constant_values = (0,0)) # add
```



## 二、超参数配置：

### 1. llm_serving

修改llama_7b_kbk_pa_dyn.yaml文件中的decode_batch_size

（1）**测试推理时延时，修改如下参数：**

  decode_batch_size: [128]

（2）**验证精度时，只能设置为1：**

  decode_batch_size: [1]

### 2. performance_serving 

修改performance_serving中的 test.sh

（1）测试推理时延时，设置为 -x 10  -T 150：

```
python test_serving_performance.py -X 10 -P 8835 -O "./" -T 150
```

备注：测试耗时时需要**将test_serving_performance_sort.py文件中的内容复制到test_serving_performance.py**

（2）验证精度时，设置为：

```
python test_serving_performance.py -X 0.1 -P 8835 -O "./" -T 5000
```

备注：验证精度时需要**将test_serving_performance_raw.py文件中的内容复制到test_serving_performance.py**



## 三、 **推理结果:**

耗时：推理1500条数据，总耗时598 s 

备注：

1）推理前必须先采取多条请求预热； 

例如运行3次以上：

curl 127.0.0.1:8835/models/llama2/generate \

-X POST \

-d '{"inputs":" I love Beijing, because","parameters":{"max_new_tokens":16, "do_sample":"False", "return_full_text":"True"}, "stream":"True"}' \

-H 'Content-Type: application/json'

2）每次推理会有1-2s的时间波动，经过多次测量，测量时间均在598-600s之间

![image-20241029210707329](C:\Users\ly\AppData\Roaming\Typora\typora-user-images\image-20241029210707329.png)



## 四、 精度验证：

500条数据精度比对均通过：

备注：

1)  验证精度时需要将llmserving中的 agent_multi_post_method_save_logits.py中的内容复制并替换agent_multi_post_method.py（记得保存）中的内容

2）注释掉model_init_multimodel.py文件中call函数增加的两行代码（514行，515行，原因上面也说了，验证精度时不能调整prefill和decode的优先级，会导致保存的顺序不一致，无法对比，但是调度是不会影响精度的，推理1500条的日志内容也完全正确）

![image-20241031171324500](C:\Users\ly\AppData\Roaming\Typora\typora-user-images\image-20241031171324500.png)

3）由于采用了perdict和postprocess异步方式推理，目前还不太稳定，测试精度时可能偶尔存在丢包的情况，这个时候请重新运行一下，以500条数据成功推理的结果为准

4)  performerce_serving 测试精度时，需要将test_serving_performance_raw.py文件中的内容复制到test_serving_performance.py 文件中，（test_serving_performance_raw.py文件中没有对input长度进行排序, 因为排序会导致文件保存顺序不一致，因此验证精度时不能排序，但是推理顺序理论上并不会影响精度）

![image-20241030110534409](C:\Users\ly\AppData\Roaming\Typora\typora-user-images\image-20241030110534409.png)



打点分析:

```
[mindformers call-chain]
-> GenerationMixin.generate                   // 循环推理，测定 25ms/token
  -> GenerationMixin.infer                    // 推理一个token单位
    -> GenerationMixin.forward (*)            // 推测 ~22.5ms
      -> LlamaForCausalLM.construct           // if prefill
      -> GenerationMixin._incremental_infer   // if decode
        -> LlamaForCausalLM.construct
    -> GenerationMixin.postprocess            // 推测 ~2.5ms

[llm-serving call-chain]
-> LLMServer.generate_answer                              // 推理一个请求单位
  -> LLMServer.register_request
    -> LLMServer.start_background_loop
      -> LLMServer.run_loop
        -> LLMServer.step
          -> AsyncMaster.step_async                       // 决定 batching 策略
            -> AsyncMaster._run_workers_async             // 测定 40ms/token -> 优化后 27ms/token
              -> Worker.predict                           // 测定 39.55ms
                - Worker._predict
                  -> DisModel.call                        // 测定 39.47ms
                    - shared_mem::write
                    - tcp::sendall
                      -> start_agent_socket_server        // 测定 63.4ms(prefill) / 37.7ms(decode)
                        - tcp::recv
                        - WorkAgent.predictc
                          - shared_mem::read
                          - agent pre-process             // 测定 12.0ms 
                          - WorkAgent.predict_for_kbk     // 测定 22.5ms
                            -> GenerationMixin.forward    // 流程同上 mindformers (*)
                          - WorkAgent.do_post_sampling    // 测定 3.04ms 
                          - shared_mem::write
                        - tcp::sendall
                    - tcp::recv
                    - shared_mem::read
              - AsyncMaster._postprocess                  // 测定 0.83ms
```



## 五、 运行环境说明：

本作品直接使用比赛说明中配置的环境，不需要安装其他环境



## 六、 代码以及npy文件路径:

**压缩包文件路径（可直接下载）**

所有文件已经打包成一个文件 file_20241031.zip，最新提交obs路径如下：

https://aics2024.obs.cn-southwest-2.myhuaweicloud.com/file_20241031.zip

其中包括llm-serving、performance_serving代码、精度验证结果file_npy文件，mindformers(这个库没有做改动，可以直接用官方的)

