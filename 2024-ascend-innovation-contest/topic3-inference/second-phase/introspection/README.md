# 大语言模型推理调优报告
## 背景简介及项目概述
大语言模型居高不下的算力消耗是制约其商业服务部署的瓶颈问题。用户访问的不可预测性对大语言模型的推理速度和并发吞吐量提出很高的要求，因此对大语言模型推理速度的优化具有重要商业和学术意义。
本项目为推理调优比赛的第二阶段，因此本报告撰写则分为以下部分，第一部分是对mindspore大模型推理算法的分析以找出主要问题，第二部分是提出新型推理调度算法提升推理速度，第三部分则总结了对代码的修改部分以及本项目的工作量，第四部分则是相应的配置文件、运行代码和结果截图，最后一部分则是对第一阶段优化算法的评估。
## 背景调研和现有模型框架分析
本项目中提供的llm服务框架类似典型的v-llm架构，本章结合代码实测结果对ms-llm-serve的算法进行分析，以找出制约推理性能的最大因素。然后本次分析均在第一阶段最优效果的基础上进行，详细改动详见最后一部分对第一阶段优化算法的评估。
大模型推理过程中任务分为prefill过程和decode过程，prefill是指输入语句首次输入大模型进行计算的过程，而decode过程则是对其进行增量推理的过程。两者最大的区别就是，prefill过程中，输入语句首次输入大模型，需要完整计算输入语句的K、V、Q，并生成第一个token，prefill阶段计算产生的KV可在随后增量推理的decode阶段重复利用（利用矩阵计算的特性），即decode阶段仅计算新生成token的KV并与之前缓存到显卡内存的KV拼接，计算第二个token，然后循环推理，直到遇到终止符，结束推理，这种利用缓存KV加速的方法即为KVcache。而缓存KV时，由于推理语句输出长度的不确定性，很难在一开始预测一个语句KV的长度，而给KV预留最大语句长度的话会造成极大的显存浪费，因此此时需要对显存进行管理，因此引入了paged attention算法对显存进行管理。
而KVcache和paged attention方法已经集成在本mindspore框架内，其直接使用底层算子实现，相应代码在mindformers/modules /paged_attention_mgr.py文件内，底层算子是P.auto_generate. PagedAttention。本项目进行优化推理时默认开启paged_attention。
大模型推理中，prefill阶段产出用户输入的第一个返回字符，decode阶段逐字符生成大模型输出。因此，通常是prefill进行一次，而decode阶段则循环推理N次（N为生成句子长度）。从迭代次数来看decode占主要，而使用KV-cache后，decode阶段的输入仅为1个token，因此通常将多个请求的token合并在一次推理中进行，即组成一个decode batch，以充分利用计算能力。如图1所示，随着batch size的增加，decode阶段的吞吐量快速上升。此时，decode阶段的吞吐量主要受限于显存，decode阶段需要存储batch内所有输入的KV，因此其最大batchsize大小与显存有关，也与生成文本的长度有关。
本项目中，能支持最大的batch为128，增加batch size之后显著降低的推理时间，由baseline的3000秒左右降低到6-700秒。



然后，将decode阶段batch size改为128后，对推理过程的输出日志进行分析，统计prefill阶段和decode阶段的时间。如下表所示，decode batch size变为128后，decode阶段耗时占比大幅下降，仅占end-to-end总时长的22%，而prefill阶段的耗时占比则为76%。是耗时的主要来源。因此，对该框架的优化应该聚焦在prefill阶段。因此下一章提出一种prefill和decode阶段的调度方法以提升prefill阶段的速度。

 | 阶段 | 	Prefill	 | Decode | 	通讯&后处理 | 
 |	--- | --- | --- | --- | 
 | 时间/占比 | 	575318ms	76%	 | 168259ms	22%	 | 15592ms	2% | 

## Prefill-Decode混合调度方法
目前业界共识一般将prefill和decode阶段当做两个完全不同的阶段处理，prefill阶段不涉及KVcache，仅有计算过程，因此被认为是计算密集型任务，而decode阶段仅对最新token计算，然而却需要调度整个batch内的KVcache，因此被认为是内存密集型任务。实际大模型框架中，通常都会对这两个进行分离处理，不会将prefill任务和decode任务放在一个batch里面。
本项目中，mindspore也是将这两个过程进行了分离处理，执行过程中通过self.prefill或is_first_iteration标识符判断是否prefill阶段。
在prefill和decode分离之后，如何安排两者之间的先后执行关系，就是推理调度方法。在mindspore中，LLM serving框架使用的是典型的prefill优先策略。即当有新的请求，而且decode batch中存在空位时，马上执行新请求的prefill阶段，然后将这单个新请求插入decode batch中，然后再进行迭代处理。这样的好处是可以将decode阶段每次迭代的batch吞吐量打满，图3是实际运行中decode每次迭代的有效token数量，可以看到整个过程中decode的每个batch都是拉满的，即每当decode中有一个请求完成输出，碰到终止符，就会马上执行一个新的prefill，将新生成的一个token加入decode过程。后面的下降是所有请求都prefill后，逐步有语句结束退出decode，然后逐渐下降到0，完成全部推理。这种模式首先每次decode迭代，完成生成并退出的语句数量都在变化，因此要执行prefill加入decode的语句数量难以固定，因此mindspore中直接将每次prefill 的语句数量写死为1，即prefill batch固定为1。（例如llm-serving/ mindspore_serving/master/master.py中的_check_prompt_predict_data_pa方法，在长度大于1直接break，甚至底层算子P.auto_generate.PagedAttention在设置phase为prefill时只输出第一个语句的logits）
 


这种调度方式，虽然可以拉满decode过程，大幅降低decode过程的时间，但是却会导致prefill阶段仅能每次迭代仅能处理单个句子。如经典的图4，虽然prefill过程随batchsize的变化通常被认为并不如decode过程显著，但如图4所示，增加prefill的batchsize也显示出了相当大的优化空间，尤其在输入语句较小的时候。
 


然后首先验证这种方法的优化效果，同时验证其泛化性，首先抛开本项目的背景，直接固定输入文本的长度，记录不同prefill batch下单个语句的执行时间。由表中可看出，提升prefill batch的优化效果还是比较明显，在较短输入文本上，执行时间甚至可以降低三倍以上。而且优化效果呈现出batch由1到2的优化效果最佳，然后随着batch的继续增加优化效果逐渐下降，其原因是随着batch的增加其逐渐达到了计算能力的瓶颈。然而将prefill batch增大，则需要牺牲一部分decode 阶段的性能，例如prefill batch为4时，decode过程就需要结束四个语句之后才能进行prefill，这个过程中，decode过程就不能拉满。很显然prefill和decode之间存在一种制约关系，增加prefill batch会减小prefill阶段时间，但会导致decode时间增加。因此需要找到两者之间的平衡达到最优，而这个最优的prefill batch大小就与decode batch直接相关。 如图5所示，对128的decode size来说，prefill batch为20时会导致仅20/128的通量损失，而对于24的decode size来说，prefill batch为20时则会导致90%的通量损失，而且这个平衡点与用户输入的语句长度也有关系。因此，现在设置一个经验系数，然后规定prefill batch size = decode batch size * 经验系数。这个经验系数即为decode过程通量牺牲的比例。



不同输入文本长度和prefill batch下单个语句的执行时间（ms）

| 输入文本长度 |	Prefill batch |	Prefill batch |	Prefill batch |	Prefill batch |
|	--- | --- | --- | --- | --- |
|		1	 |	2 |		4	 |	8	 |	16 |	
|	128	 |	35	 |	21	 |	17	 |	13 |		11
|	256	 |	44 |		31 |		23 |		21 |		20 |	
|	512 |		63	 |	47	 |	44	 |	42	 |	41 |	
|	1024 |		93	 |	84 |		83 |		83 |		82 |	




然而在本项目中，prefill阶段推理耗时占比在76%左右，根据以上的验证结果，增加prefill batch size带来的好处远大于对decode过程的不利影响。然后，如表中所示，batch size由1变为2时，对推理速度的提升最大，而且本项目框架中由于静态图编译特性，在prefill batch size过大时，会出现在1到batch size之间的静态图编译过程，因此最后选择经验系数为1/64，即prefill batch size为2. 最后的推理调度过程如图所示
 

## 优化验证和运行参数
超参配置：
根据一阶段结果，在原有超参上，将decode_batch_size调整为128，此处prefill_batch_size 为无效参数，因此仍保持为1,相应配置文件为llm-serving/configs/llama/llama_7b_kbk_pa_dyn.yaml
测速结果：
此时1500条数据的推理时间为436.9s，相关日志为performance_serving/test_1.log
 
日志中文字输出与原输出进行了对比，输出完全一致。
test.sh设置为python test_serving_performance.py -X 10 -P 8835 -O "./" -T 150

精度测试：
精度测试与第一阶段一致，将配置中decode_batch_size改回1（与原有配置相同），配置文件为：llm-serving/configs/llama/llama_7b_kbk_pa_dyn_val.yaml
日志文件为：
performance_serving/test_val.log
验证结果如下图，通过测试：
测试得到的npy文件在下面链接中：
https://gf-finetune1.obs.cn-southwest-2.myhuaweicloud.com/fintune/file_npy.zip

 
运行说明：
运行环境与官方环境相同，所有代码包括日志的压缩包如下：
https://gf-finetune1.obs.cn-southwest-2.myhuaweicloud.com/fintune/llm-serving.zip
https://gf-finetune1.obs.cn-southwest-2.myhuaweicloud.com/fintune/mindformers.zip
https://gf-finetune1.obs.cn-southwest-2.myhuaweicloud.com/fintune/performance_serving.zip

为实现prefill多batch，以上三个包均有修改，其全部在在work根目录解压后运行，根据经验系数对prefill batch size的调整也集成在了代码内，其中checkpoint_download中需要重下权重文件，后续运行命令均与官方相同，测试精度时请用llm-serving/configs/llama/llama_7b_kbk_pa_dyn_val.yaml配置文件
## 代码修改
由于Prefill batch在底层代码上写死为1，因此实现该功能涉及到了多处对原有代码框架的修改，需要对原有推理框架进行完整解读。以下列出代码主要修改的地方：
llm-serving/mindspore_serving/agent/agent_multi_post_method.py
predict方法中prefill 阶段current index的处理逻辑修改
```seq_length = input_ids.shape[1]
current_index = [
    current_index[i] - i * seq_length
    for i in range(current_index)
]
predict方法中索引修改，原有代码直接取索引0
logging.debug(f'decode index {decode_index}')
for idx, index in enumerate(decode_index):
    index = int(index)
    logging.debug(f"===============predict self.kbk_targets len:{len(self.kbk_targets)}")
    logging.debug(f"===============predict decode index:{index}")
    if self.is_prefill:
        logging.debug("kbk predict prefill start.")
        self.kbk_targets[index] = input_ids[idx]
```
predict_for_kbk 方法重写，原有框架中算子规定prefill阶段仅处理第一个语句，因此本方法必须重写，未重写本方法的语句输出则完全是错误的！
```
def predict_for_kbk(self, current_index, input_ids, valid_length, block_tables, slot_mapping):
    # 封装调用模型参数
    model_kwargs = {"current_index": current_index}
    model_inputs = self.mindspore_model.prepare_inputs_for_generation(input_ids, **model_kwargs)
    logging.debug(f"predict model_inputs value is {model_inputs}")
    # 调用mindformers进行推理
    predict_time = time.time()
    if self.mindspore_model.config.use_past:
        logging.debug(f"predict before pa predict_for_kbk. first phase {self.mindspore_model.phase}")
        if self.is_prefill:
            self.mindspore_model.is_first_iteration = True
            self.mindspore_model.phase = "predict"
            self.mindspore_model.add_flags_custom(is_first_iteration=True)

            batch_size = input_ids.shape[0]
            seq_length = input_ids.shape[1]
            current_index2 = [
                valid_length[i] - 1 + i * seq_length
                for i in range(batch_size)
            ]

            res = self.mindspore_model(input_ids=Tensor(input_ids, mstype.int32),input_position=Tensor(current_index2, mstype.int32),init_reset=Tensor([False], mstype.bool_)
                                   ,batch_valid_length=Tensor([valid_length], mstype.int32),block_tables= Tensor(block_tables, mstype.int32),slot_mapping=Tensor(slot_mapping, mstype.int32))
            mindspore.hal.synchronize()
            self.mindspore_model.is_first_iteration = False
            self.mindspore_model.phase = "increment"
            # first iter done, go to other iters
            self.mindspore_model.add_flags_custom(is_first_iteration=False)

        else:

            res, current_index,model_input,org_input = self.mindspore_model.forward(input_ids=input_ids,
                                                valid_length_each_example=valid_length,
                                                generation_config=self.mindspore_model.config,
                                                block_tables=block_tables,
                                                slot_mapping=slot_mapping,
                                                prefill=self.is_prefill,
                                                **model_kwargs)
    else:
        res = self.mindspore_model(**model_inputs)
    logging.info('predict time is {} is prefill {}'.format((time.time() - predict_time) * 1000,self.is_prefill))
    logging.info("mindspore_model res : %s;", res)
    logging.info("mindspore_model input : %s;", model_input)
    outputs = res[0] if isinstance(res, tuple) else res

    return outputs
```
该文件还有argmax函数numpy化的相关代码可以提升几秒的速度，此处不在赘述。
```
llm-serving/mindspore_serving/master/master.py
_check_prompt_predict_data_pa方法中修改post逻辑
for index, item in enumerate(entry_metadata_list):
    if not item.is_prompt or item.entry_data.status == EntryStatus.INPUT_OUTOFRANGE:
        # logging.debug(f"GFtest: item {item.is_prompt} {item.entry_data.status == EntryStatus.INPUT_OUTOFRANGE} index {index}")
        continue
    # input_entry_metadata_list = [item]
    # index_list = [index]
    input_entry_metadata_list.append(item)
    index_list.append(index)
    if len(index_list)>batch_size//32:
        break

llm-serving\mindspore_serving\schedule\schedule.py
try_substitute_entry修改prefill插入decode阶段的逻辑，改为达到prefill batch size后再插入
mindformers/mindformers/generation/text_generator.py
mindformers/mindformers/models/llama/llama.py
```
以上代码中还有部分调试代码，此处不再赘述





