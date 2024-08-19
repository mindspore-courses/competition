# moon选手的作品报告

## 业界推理优化算法调研

### PagedAttention
PagedAttention 是一种受到操作系统中的虚拟内存和分页技术启发的注意力算法。传统的Transformer模型中的注意力机制通常将键值（KV）缓存连续存储在内存中，这可能导致内存碎片和低效使用。

PagedAttention 通过将键值缓存分割成固定大小的块，并允许这些块存储在非连续的内存空间中，来解决这些问题。这种方法有助于减少内存浪费并提高内存效率，特别是在处理大模型和长序列时。

主要优势：

1. 内存效率：通过在块级别而不是序列级别管理内存，PagedAttention减少了内存碎片和浪费，在某些情况下实现了接近零的内存浪费 ([Sky_CS_Berkeley](https://sky.cs.berkeley.edu/project/vllm/))([vLLM Blog](https://blog.vllm.ai/2023/06/20/vllm.html))。
2. 吞吐量提升：这种高效的内存管理允许更多的请求一起批处理，从而显著提高吞吐量。例如，利用PagedAttention的vLLM相比最先进的系统如FasterTransformer和Orca显示了2-4倍的吞吐量提升([Sky_CS_Berkeley](https://sky.cs.berkeley.edu/project/vllm/)) ([PyCon India](https://in.pycon.org/cfp/2024/proposals/revolutionizing-llm-serving-pagedattention-and-vllm-for-unmatched-throughput-efficiency-and-seamless-integration-with-popular-hugging-face-models%7EaxkVr/))。
3. 灵活的内存共享：PagedAttention使请求之间的高效内存共享成为可能，这在并行采样和束搜索等场景中尤为有用，减少内存开销高达55%，并提高吞吐量高达2.2倍 ([vLLM Blog](https://blog.vllm.ai/2023/06/20/vllm.html))。

### 持续批处理
持续批处理优化了LLM服务系统处理传入请求的方式。传统的批处理方法由于请求的到达时间和输入输出序列长度的不同，往往导致效率低下。持续批处理（也称为细粒度或迭代级批处理）通过在迭代级而不是请求级操作来解决这些问题。

主要优势：

1. 减少延迟：持续批处理允许新请求在每次迭代后进行处理，而不是等待整个批次完成，从而显著减少排队延迟 ([ar5iv](https://ar5iv.labs.arxiv.org/html/2309.06180))。
2. 资源利用率提高：通过消除将输入和输出填充到相同长度的需求，持续批处理更有效地利用GPU资源，提高整体吞吐量 ([ar5iv](https://ar5iv.labs.arxiv.org/html/2309.06180))。
3. 更高的吞吐量：这种方法允许请求的动态高效批处理，这对于在高需求场景中保持高吞吐量至关重要 ([GitHub](https://github.com/vllm-project/vllm))。

## 本作品使用的推理优化算法介绍
### 后处理耗时优化方案：
由于后处理统一使用贪婪搜素策略，即求32000词表中的最大值的id，baseline做法用的是调用ms的ops.Argmax算子：

```
class ArgmaxPost(nn.Cell):
    def __init__(self):
        super(ArgmaxPost, self).__init__()
        self.argmax = ops.Argmax(output_type=ms.int32)
        # self.reshape = ops.reshape()
    def construct(self, x):
        x = ops.reshape(x, (x.shape[0], x.shape[-1]))
        output = self.argmax(x)
        return output
```
耗时测量：

首先在do_post_sampling函数位置，使用time函数计算执行计算的耗时。计时区间为进行argmax计算的全过程，包括后面转为numpy类型的时间。

```

start_time = time.time()  
logging.info("do_post_sampling outputs_np type is f, value is {}".format(outputs_np.dtype, outputs_np))  
  
do_sample = self.get_consistent_batch(decode_index)  
  
if self.config.model_config.backend == "ge":  
    if self.config.serving_config.enable_host_post_sampling:  
        if not do_sample:  
            target = self._post_sampling_argmax_host(outputs_np)  
            target = target.reshape((self.current_batch_size,))  
            target = np.squeeze(target, axis=1)  
        else:  
            target = self._post_sampling_topk_host(outputs_np, decode_index, prefill) 
    else:  
        if not do_sample:  
            target = self._post_sampling_argmax_npu(outputs_np)  
        else:  
            target = self._post_sampling_topk_npu(outputs_np, decode_index, prefill)  
else:  
    if not do_sample:  
        self.targets.clear()  
        target = self.argmax_model(outputs_np)  
    else:  
        target = self._post_sampling_topk_kbk(outputs_np, decode_index) 

    if isinstance(target, Tensor):  
        target = target.asnumpy()
    output_info = outputs_np.asnumpy()
  
logging.info('argmax_model time is {} '.format((time.time() - start_time) * 1000)) 

```

改进：后处理耗时太长，改写算法，直接用numpy实现：

```
# 新的 ArgmaxPost 类（使用 numpy 实现）
class ArgmaxPost:
    def __init__(self):
        pass
    def construct(self, x):
        x = x.reshape((x.shape[0], x.shape[-1]))
        output = np.argmax(x, axis=-1)
        return output

self.argmax_model = ArgmaxPost()

def do_post_sampling(self, outputs_np, outputs_shm, output_logprob_shm,decode_index, prefill=True):
    start_time = time.time()
    # 确保 outputs_np 是 numpy 数组
    if isinstance(outputs_np, Tensor):
        outputs_np = outputs_np.asnumpy()
    logging.info("do_post_sampling outputs_np shape is {}, value is{}".format(outputs_np.shape, outputs_np))
    do_sample = self.get_consistent_batch(decode_index)
    if self.config.model_config.backend == "ge":
        if self.config.serving_config.enable_host_post_sampling:
            if not do_sample:
                target = self._post_sampling_argmax_host(outputs_np)
                target.reshape((self.current_batch_size,))
                target = np.squeeze(target, axis=1)
            else:
                target = self._post_sampling_topk_host(outputs_np,decode_index, prefill)
        else:
            if not do_sample:
                target = self._post_sampling_argmax_npu(outputs_np)
            else:
                target = self._post_sampling_topk_npu(outputs_np,decode_index, prefill)
        output_info = outputs_np # 假设 get_data_to_numpy 返回 numpy 数组
    else:
        if not do_sample:
            self.targets.clear()
            #logging.info("pre outputs_np shape is {}, value is{}".format(outputs_np.shape, outputs_np))
            #logging.info("outputs_np type: {}".format(type(outputs_np)))
            #logging.info("outputs_np dtype: {}".format(outputs_np.dtype))
            #logging.info("outputs_np shape: {}".format(outputs_np.shape))
            target = self.argmax_model.construct(outputs_np)
        else:
            target = self._post_sampling_topk_kbk(outputs_np, decode_index)
        if isinstance(target, np.ndarray):
            target = target
        output_info = outputs_np
    logging.error('argmax_model time is {}'.format((time.time() - start_time) * 1000))
    logging.info("do_post_sampling target type is {}, value is{}".format(target.dtype, target))
    logging.info("do_post_sampling output_info type is {}, value is{}".format(output_info.dtype, output_info))
    if self.rank_id == 0:
        if prefill:
            for index in decode_index:
                tmp = np.ndarray((index + self.current_batch_size,),dtype=target.dtype, buffer=outputs_shm.buf)
                tmp[index: index + self.current_batch_size] = target[:]
                logprob_list = []
                for idx, tag in enumerate(target):
                    logprob_list.append(output_info[idx][int(tag)])
                tmp_logprob = np.ndarray((index + self.current_batch_size,),dtype=np.float64,
                buffer=output_logprob_shm.buf)
                tmp_logprob[index: index + self.current_batch_size] = logprob_list[:]
                self.targets[index: index + self.current_batch_size] = target[:]
        else:
            tmp = np.ndarray((self.current_batch_size,), dtype=target.dtype,
            buffer=outputs_shm.buf)
            tmp[:] = target[:]
            logprob_list = []
            for idx, tag in enumerate(target):
                if len(output_info.shape) == 2:
                logprob_list.append(output_info[idx][int(tag)])
                else:
                logprob_list.append(output_info[idx][0][int(tag)])
            tmp_logprob = np.ndarray((self.current_batch_size,),dtype=np.float64, buffer=output_logprob_shm.buf)
            tmp_logprob[:] = logprob_list[:]
            self.targets[:] = target[:]

```

改进前后耗时比对：

原始 后处理耗时：

```

argmax model time is 1.8422603607177734
argmax_model time is 1.566171646118164 
argmax model time is 1.5742778778076172 
argmax model time is 1.5418529510498047c
argmax model time is 1.5556812286376953 
argmax model time is 1.5604496002197266 
argmax model time is 1.5561580657958984
argmax_model time is 1.5518665313720703 
argmax model time is 1.5592575073242188
argmax model time is 1.5444755554199219c
argmax model time is 1.581430435180664
argmax model time is 1.5611648559570312
argmax_model time is 1.5447139739990234
argmax model time is 1.54876708984375
argmax model time is 1.567840576171875
argmax model time is 1.6522407531738281
argmax model time is 1.5442371368408203 
argmax model time is 1.560211181640625
argmax_model time is 1.5490055084228516 
argmax model time is 1.5635490417480469
argmax_model time is 1.7533302307128906
argmax model time is 1.603841781616211
argmax model time is 1.5835762023925781 
argmax model time is 1.5740394592285156c
argmax model time is 1.5702247619628906
argmax model time is 1.6927719116210938
argmax_model time is 1.6138553619384766
```

改进后耗时：

```
argmax model time is 0.5939006805419922
argmax_model time is 0.5891323089599609
argmax_model time is 0.6508827209472656
argmax model time is 0.7717609405517578
argmax model time is 0.6158351898193359
argmax model time is 0.5974769592285156
argmax model time is 0.5955696105957031
argmax_model time is 0.6136894226074219
argmax_model time is 0.5974769592285156
argmax_model time is 0.6153583526611328
argmax_model time is 0.5946159362792969
argmax_model time is 0.5936622619628906
argmax_model time is 0.6301403045654297
argmax_model time is 0.5931854248046875 
argmax_model time is 0.5967617034912109
argmax model time is 0.6160736083984375 
argmax_model time is 0.6017684936523438
argmax_model time is 0.5929470062255859
argmax model time is 9.6163120269775391
argmax model time is 0.6613731384277344 
argmax model time is 0.6325244903564453 
argmax_model time is 0.6570816040039062
argmax model time is 0.5934238433837891 
argmax_model time is 0.5998611450195312
argmax_model time is 0.5922317504882812
argmax model time is 0.5967617034912109
argmax model time is 0.6792545318603516 
argmax model time is 0.7989406585693359
argmax_model time is 0.6437301635742188 
argmax_model time is 0.6518363952636719
argmax_model time is 0.6122589111328125

```

后处理耗时由原来的1.6s降低至目前0.6s

原始实现耗时原因分析：对32000个浮点数求最大值的操作是memory bound类型的，推理过程中对于显存带宽的压力本身就很大，再将数据输入到NPU中进行比较，导致耗时严重。

先将output数据转为numpy类型，直接使用CPU进行求解最大值的操作，合理重叠了NPU运算和数据传输的过程，求解出词表中概率的最大值。

这个改进在单batch推理下可以得出正确答案，但是在多batch推理中，会有一些问题，导致推理结果错乱，所以目前还存在bug，后续看情况修复一下。由于并不完善后续的测速方案中并没有使用这个优化方法进行测速。

### seq_len调整

根据观察，请求的输出字段长度并没有特别长，大部分在100-500之间。在日志中找到1个较长的输出，
例如：

```
2024-07-29 16:14:17,842 - test_llama.log - INFO - {'input': 'Generate a list of
elements in a periodic table.', 'resp_text': '\nGenerate a list of elements in a
periodic table. The elements are sorted by atomic number.\nThe elements are
sorted by atomic number.\nThe elements are sorted by atomic number. The elements
are sorted by atomic number.\nThe elements are sorted by atomic number. The
elements are sorted by atomic number. The elements are sorted by atomic
number.\nThe elements are sorted by atomic number. The elements are sorted by
atomic number. The elements are sorted by atomic number. The elements are sorted
by atomic number. The elements are sorted by atomic number. The elements are
sorted by atomic number. The elements are sorted by atomic number. The elements
are sorted by atomic number. The elements are sorted by atomic number. The
elements are sorted by atomic number. The elements are sorted by atomic number.
The elements are sorted by atomic number. The elements are sorted by atomic
number. The elements are sorted by atomic number. The elements are sorted by
atomic number. The elements are sorted by atomic number. The elements are sorted
by atomic number. The elements are sorted by atomic number. The elements are
sorted by atomic number. The elements are sorted by atomic number. The elements
are sorted by atomic number. The elements are sorted by atomic number. The
elements are sorted by atomic number. The elements are sorted by atomic number.
The elements are sorted by atomic number. The elements are sorted by atomic
number. The elements are sorted by atomic number. The elements are sorted by
atomic number. The elements are sorted by atomic number. The elements are sorted
by atomic number. The elements are sorted by atomic number. The elements are
sorted by atomic number. The elements are sorted by atomic number. The elements
are sorted by atomic number. The elements are sorted by atomic number. The
elements are sorted by atomic number. The elements are sorted by atomic number.
The elements are sorted by atomic number. The elements are sorted by atomic
number. The', 'res_time': 421.4911723136902, 'first_token_time':
303.11994767189026}

```

这个句子的输出是比较长的，可以调用tokenizer分词查看句子的长度，或者直接通过huggingface在线的分词器来看这个句子的分词结果由图可知，整个json字段的token数才不到五百，所以将推理的句子长度配置设置为600，必然能保证全部推理任务的正常进行，减少句子长度还会节约大量的显存占用。


![image]()

修改句子输出长度可以提升推理系统的吞吐量，但是会导致词表概率发生一些变化，无法通过校验，所以没办法使用这种策略。

## 超参配置介绍

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

```
python test_serving_performance.py -X 10 -P 8835 -O "./" -T 150
```

在llm-serving启动完毕后，先执行单条或者多条推理请求预热，后续测速结果更加稳定。


## 提交推理的日志、配置文件
test_performance_2024-07-31-00_20.log

OBS：https://mooninfer.obs.cn-southwest-2.myhuaweicloud.com/moon.zip

## 源码包

OBS：https://mooninfer.obs.cn-southwest-2.myhuaweicloud.com/moon.zip






