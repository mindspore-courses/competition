## 作品介绍

团队名：勇敢牛牛yyds

队长：刘洋

联系方式：18707311330

提交时间：20240729

### 一、优化策略
本作品主要从调度策略进行优化，主要修改均在llm-serving框架中

#### 1.1 加入时间间隔阈值

通过阅读llm-serving框架源码发现，在schedule.py的_continuous_batch_pa(）函数中，每个step后更
新batch数据时， 会不断查看waiting_request_queue有没有等待的请求，一旦有等待的请求，便直接
插入空槽位。这样做有一个问题，因为一旦发生替换，就必须暂停running_request_list中的request的
decode，需要先完成插入进来的request的prefill（因为模型在一个阶段要么只进行decode，要么只进
行prefill）。如果状态切换过于频繁，对于整体的推理时延并不是最优，因此可以参考vllm设置时间间
隔阈值。

- 调度间隔设置得太小，每次调度都只关心waiting中的新请求，这样发送旧请求的用户就迟迟得不
到反馈结果。且此时waiting队列中积累的新请求数量可能比较少，不利于做batching，浪费了并
发处理的能力。
- 调度间隔设置得太大，waiting中的请求持续挤压，同样对推理的整体吞吐有影响。


因此，主要在schedule.py文件中进行如下修改()：

```
### 增加_passed_delay，用于确定当前是否满足时间阙值  
def _passed_delay(self, now: float) -> bool:  
    ### 上次调度的时间距离当前时间的时延，如果上次没有发生调度，那就不用更改  
    ### 不然如果一直没有调度，这个值  
    if self.prev_prompt:  
        self.last_prompt_latency = now - self.prev_time  
        ### 这里直接记录为now也没问题，因为如果这次没有调度，后面也不会用到这个值；  
        #### 这样做也可以避免self.last_prompt_latency变成一个很大的值，不然  
        # passed_delay会一直为false  
    self.prev_time = now  
    self.prev_prompt = False  
  
    # Delay scheduling prompts to let waiting queue fill up  
    if self.delay_factor > 0:  
        earliest_arrival_time = min([e.arrival_time for e in self.waiting_request_queue])  
        passed_delay = (  
            (now - earliest_arrival_time) >  
            (self.delay_factor * self.last_prompt_latency)  
        )  
    else:  
        passed_delay = True  
    return passed_delay

# 增加时间判断  
def _continuous_batch_pa(self):  
    ServingBlockMemPool.instance().reset_budget()
    # ServingBlockMemPool.instance().log_status() 
    # 这里只是用padding先凑成batch，一开始里面的请求都是invalid  
    self.try_initialize_paddings_pa()  
    # self.log_running_list("schedule start running status")  
    # 判断batch内的running entry，能否进行本轮推理?  
    num_entry_swapped_out = 0  
    while not self.can_predict_current_batch():  
        # 如果不能，swap出去已有请求，使用padding（无效请求）替代  
        self.reset_all_budgets()
        self.try_swap_valid_entries()  
        num_entry_swapped_out += 1  
    if num_entry_swapped_out:  
        self.reset_all_budgets()
        return 
    # 3. 处理新请求
    # logging.debug("determine if can process new request...")
    ### 只要有等待的请求，就去尝试替换出 running_request_list中的invalid请求（即填充的无效请求）
    ### 针对waiting_request_queue中的请求的执行顺序，是否可以根据其输入长度，越短的优先
    ### 这一步可以优化，参考vllm，不要一有空余槽位和等待的请求就将其加入 running，
    ### 因为一旦加入新的序列，就要停止当前running中的请求的decode，转而进行新加入请求的prefill
    ### 这样整体的吞吐不一定最优，应该设置一个时间阈值，当大于这个阈值时再进行加入
    # 根据_passed_delay判断这次step要不要加入新的原生promt

    if self._passed_delay(time.time()):
        while self.waiting_request_queue:  
            # 如果有空batch槽，尝试插入  
            # logging.debug("has new entry, trying to enter current batch")
            if not self.try_substitute_entry(): 
                # 尝试失败，退出  
                break  
            else:  
                ############# add  
                self.prev_time = time.time()  # 记录这次的调度时间  
                self.prev_prompt = True  # 记录已经调度过  
                # #######  
    self.reset_all_budgets()   
    # ServingBlockMemPool.instance().log_status()

```

同时，entry.py文件中，增加 到达时间参数arrival_time


```
class EntryMetaData:  
    """
        entry meta used in combine batch  
    """  
    def __init__(self, arrival_time: float, page_attention: bool, request_id: str, is_prompt: bool, entry_data: EntryData,, entry_id: int, prompt: str, block_size： int) -> None:  
        self.arrival_time = arrival_time  
        self.request_id = request_id  
        self.is_prompt = is_prompt  
        self.entry_data = entry_data 
        self.entry_id = entry_id  
        self.prompt = prompt  

        if page_attention: 
            self.cache_engine = ServingCacheEngine(block_size=block_size,  
                                                   pool=ServingBlockMemPool.instance())
 
    def get_prompt(self) -> str:  
        return self.prompt  
  
    def get_entry_id(self) -> int:  
        return self.entry_id  
  
    def get_entry_data(self) -> EntryData:
        return self.entry_data  
  
    def get_token(self):  
        """get token of a request used to conduct inference"""
        return self.entry_data.get_all_tokens()  
  
    def get_infer_stage(self):  
        return self.is_prompt  
  
    def set_is_prompt(self, statue: bool) -> None:
        self.is_prompt = statue

```


#### 1.2 优化抢占策略
抢占策略的实现主要在try_swap_valid_entries( )函数中，目前存在如下问题：

swap下来的valid entries会放入 waiting_request_queue的最前端，下一次则优先级最高，这样有一个
问题，当发生多次swap时，有多个推理仅完成一部分的request不断的插入到waiting队列的最左边，最
先插入的request相对于最后插入的request的优先级反而更低，对于被swap下来的request, 其优先级
策略可以优化。

优化方案：使用单独的队列（swapped_request_queue）存放swap下来的valid entries，并且优先调
度swapped_request_queue中的request

代码修改如下：

1) 代码中增加 self.swapped_request_queue的初始化

```
class Schedule:

    """static batch strategy"""
    def __init__(self, config: ServingConfig):
        self.swapped_request_queue: Deque[EntryMetaData] = Deque([])  # add  
        self.waiting_request_queue: Deque[EntryMetaData] = Deque([])  
        self.running_request_list: List[EntryMetaData] = []  
        self.count_of_invalid_sample = 0  
        self.config = config  
        self.batch_size = config.model_config.decode_batch_size[0]  
        self.eos_token = config.model_config.end_token  
        self.batch_waiting_time = config.serving_config.prefil1_batch_waiting_time
        self.decode_batch_waiting_time = config.serving_config.decode_batch_waiting_time 
        self.batching_strategy = config.model_config.batching_strategy  
        self.max_input_len = config.model_config.seq_length[-1] if len(config.model_config.seq_length) > 0 else 4096  
        # batch中有效token的最大index，初始化为-1  
        self.max_valid_index = -1  
        self.dyn_batch = config.model_config.decode_batch_size  
        # add for_passed_delay  
        self.prev_prompt = False # 用于判断之前是否发生过从waiting队列调度到running序
        self.prev_time = 0.0
        # Last prompt step's latency
        self.last_prompt_latency = 0.0  
        self.delay_factor = 0.5

```

2) 将swap下来的valid entries放入单独的队列

```


```














