# 昇思MindSpore模型开发挑战赛【推理调优赛题】

- 团队：向日葵
- 存储链接：https://sunflower-game.obs.cn-southwest-2.myhuaweicloud.com/%E5%90%91%E6%97%A5%E8%91%B5.zip

# 1. 业界推理优化算法调研

### Attention优化

- Flash attention

Flash attention深度优化了cuda高速显存访问次数，从而可以数倍提升推理和训练速度。本来原始的self attention计算保存O(n^2)的中间结果。Flash attention把计算拆成很多小block，一个block一个block的计算，内存占用降低到一个block的大小。

- Page attention

因为块在内存中不需要连续，因而可以用一种更加灵活的方式管理 key 和 value ，就像在操作系统的虚拟内存中一样：可以将块视为页面，将 token 视为字节，将序列视为进程。序列的连续逻辑块通过块表映射到非连续物理块中。物理块在生成新 token 时按需分配。

### Batch

服务相关优化主要包括 Continuous Batching、Dynamic Batching 和 异步 Tokenize / Detokenize。其中 Continuous Batching 和 Dynamic Batching 主要围绕提高可并发的 batchsize 来提高吞吐量，异步 Tokenize / Detokenize 则通过多线程方式将 Tokenize / Detokenize 执行与模型推理过程时间交叠，实现降低时延目的。

### **低比特量化**

回归到 LLM 模型推理吞吐量和时延这两个重要的性能指标上：吞吐量的提升主要受制于显存容量，如果降低推理时显存占用量，就可以运行更大的 batchsize，即可提升吞吐量；LLM 推理具有 Memory-bound 特点，如果降低访存量，将在吞吐量和时延两个性能指标上都有收益。低比特量化技术可以降低显存占用量和访存量，其能取得加速的关键在于显存量和访存量的节省以及量化计算的加速远大于反量化带来的额外开销。

### 投机

投机采样（Speculative decoding，[FlexFlow](https://github.com/flexflow/FlexFlow)）针对 LLM 推理串行解码特点，通过引入一个近似模型来执行串行解码，原始模型执行并行评估采样，通过近似模型和原始模型的互相配合，在保证精度一致性的同时降低了大模型串行解码的次数，进而降低了推理时延。因为拒绝采样是有overhead的，大小模型差异太大拒绝次数过多，大小模型速度本身差别不大，推理runtime实现等因素都会导致观察不到加速效果。

# 2. 本作品使用的推理优化算法介绍

在llm-serving项目中，执行predict后再顺序执行_postprocess，其中_postprocess包含对生成的token进行处理以及将token转换为字符（detokenizer），而接下来的生成步骤（predict）不依赖 detokenizer，因此可以将其detokenizer与原流程进行解耦，使之流水线化。

[llm-serving workflow](https://github.com/JoegameZhou/competition/blob/master/2024-ascend-innovation-contest/topic3-inference/first-phase/assets/workflow.svg)

模型推理和detokenizer异步进行，从而大大提高 GPU 的利用率，增加推理速度。

[Pipeline](https://github.com/JoegameZhou/competition/blob/master/2024-ascend-innovation-contest/topic3-inference/first-phase/assets/pipeline.svg)



具体实现方式如下：

1. 为了管理队列传输的数据，在master.py定义一个BatchTokenIdOut类

```python
class BatchTokenIdOut:
    def __init__(self,req_batch_id:int = None,output:str = None,entry_metadata_list = None,index_list = None, skip_inference = False):
        self.req_batch_id = req_batch_id
        self.output = output
        self.entry_metadata_list = entry_metadata_list 
        self.index_list = index_list
        self.skip_inference = skip_inference
```

1. 在class AsyncMaster初始化中定义一个detokenize队列

```python
class AsyncMaster(Master):
    def __init__(
        self,
        config: ServingConfig,
        request_engine: RequestEngine
    ):
        super().__init__(config)
        self.detokenizer_que = asyncio.Queue()
        self.request_engine = request_engine
```

1. 定义一个往队列里发送detokenizer请求的函数

```python
def send_post_process(self,output,entry_metadata_list,index_list,skip_inference=False):
        self.detokenizer_que.put_nowait(
            BatchTokenIdOut(self.req_batch_id,output,entry_metadata_list,index_list,skip_inference)
        )
```

1. 在 async def _run_workers_async 中将所有self._postprocess操作换成上面的send_post_process函数

```python
 async def _run_workers_async(self, current_batch_size, entry_metadata_list):
	 ...
	 if len(prompt_token_empty_list) > 0:
		 self.send_post_process([INPUT_EMPTY_TOKEN], entry_metadata_list=entry_metadata_list,
	                                     index_list=prompt_token_empty_list,
	                                     skip_inference=True)
	 ...
	 if len(out_of_range_index_list) > 0:
	    self.send_post_process([INPUT_OUT_OF_TOKEN], entry_metadata_list=entry_metadata_list,
	                             index_list=out_of_range_index_list,
	                             skip_inference=True)
	 ...                                  
	 self.send_post_process(output, entry_metadata_list=entry_metadata_list, index_list=index_list)
	 ...
 
```

1. 在队列里接收需要detokenizer的请求数据

```python
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

1. 在llm_server_post.py中建立一个异步循环

```python
class LLMServer:
	def start_background_loop(self) -> None:
	    # todo
	    self.status = 1
	    """Start the background loop."""
	    if self.is_running:
	        raise RuntimeError("Background loop is already running.")
	    self.background_loop = asyncio.get_event_loop().create_task(self.run_loop())
	    asyncio.get_event_loop().create_task(self.master.handle_detokenization_loop())
```

# 3. 超参配置介绍

### 超参数概览
本作品在调优过程中进行了多轮测试，尝试对 "llama_7b" 模型的关键超参数进行优化，以下是最终的超参配置。

```yaml
model_config:
  model_name: 'llama_7b'
  max_generate_length: 4096
  end_token: 2
  seq_length: [4096]
  vocab_size: 32000
  prefill_batch_size: [128]
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

### 关键超参数分析

##### 1. max_generate_length: 4096
此长度设置允许模型生成长度达到4096个令牌，帮助模型在不牺牲文本连贯性和信息完整性的前提下，达到较高的输出质量。

##### 2. prefill_batch_size: [128] & decode_batch_size: [128] 
测试推理时常时**将两个批处理大小设为128**，使模型能够更有效地利用计算资源，提高数据处理的并行度，在保持输出质量的同时，提高了整体的处理速度。测试精度时batch_size都改为1。

##### 3. zactivate_len: [512, 1024, 2048, 4096]
通过分段激活，模型能更高效地处理不同长度的输入，优化计算资源的分配。

##### 4. batching_strategy: 'continuous'
策略适用于处理高并发请求，在需求量大且持续的情况下能显著减少等待时间并提升响应速度。

##### 5. page_attention: True
启用分页注意力机制允许模型在处理大型序列，可以有效地分配计算资源，减少因长序列处理引起的延迟。

# 4. 优化后的推理总时长

在 -X 和 -T 的设置为6和250的情况下，推理总时长为790.90s
# 5. 运行环境说明

无改动，与实验指导手册中提供的环境配置相同。
```
# 启动服务
cd /home/ma-user/work/llm-serving/
python examples/start.py --config /home/ma-user/work/llm-serving/configs/llama/llama_7b_kbk_pa_dyn.yaml
```
```

# 启动推理请求
cd /home/ma-user/work/performance_serving
nohup sh test.sh > test_sh.log &
```
其中测试精度使用batch_size 为1 的配置，测试时长使用batch_size为128的配置



