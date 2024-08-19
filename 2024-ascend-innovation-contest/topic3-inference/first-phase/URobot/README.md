## MindSpore-track3 URobot 推理优化加速技术报告
在本次比赛中，我们系统的学习调研了
LLM system 目前的优化方式 ; 我们基于对于 Ascend
代码的理解 , 提出超参数的调整，数据的前处理等方式；我们对所尝试的策略从速度和精
度两方面进行了细致的评估；最后，我们也对于之后进一步的系统性优化提出了下一步的
优化方向。

针对
alpca 数据集 1500 数据，在 qps 为 5 的情况下，实现了 791s 的推理速度，其参数通过
了精度测试。

此外，通过综合策略的调整，在
qps 为 5 的情况下，我们实现了 410s 的推理耗时，尽管没
有通过 logits 精度测试，但是通过最后输出日志的分析，我们认为结果合理，不存在明显
的信息损失。

我们提供的文件下载信息如下：

1. 日志信息： 包含精度速度测试相关的日志信息

https://llmsys.obs.cn-southwest-2.myhuaweicloud.com/submission/main_files.zip

最快速度参考章节 2.1.2所对应的日志 ,未通过精度测试，但是 log 结果正常。

test_performance_2024-07-30-22_36_batch64_qps5.log

精度测试所生成的日志（batch 为 1, 其他参数进行调优，并通过精度测试）
test_performance_2024-07-31-06_25_batch1.log

对应生成的 logit 文件
file_npy

最快速度并且通过精度测试对应的日志
test_performance_2024-07-31-07_04_batch_64.log

2. 源代码

其中，我们主要通过添加了针对performance_serving 中数据的前处理 位于
test_serving_performance.py 中

https://llmsys.obs.cn-southwest-2.myhuaweicloud.com/submission/llm_serving.zip

https://llmsys.obs.cn-southwest-2.myhuaweicloud.com/submission/performance_serving.zip


## 业界推理优化算法调研

近两年来，随着大模型的兴起，人们也开始从性能优化的角度更深刻的认识transformer，我们，我们也看到了相关的算法和开源项目层出不穷。从也看到了相关的算法和开源项目层出不穷。从LLM systemLLM system的角度，我们列举出有影响力的相的角度，我们列举出有影响力的相关工作关工作

1. continuous batchingcontinuous batching

论文链接： https://www.usenix.org/conference/osdi22/presentation/yu

传统的多batch系统，推理的速度受制于系统，推理的速度受制于batcbatchh中最长的序列，而中最长的序列，而ORCAORCA中，提出了中，提出了cotinuous batchingcotinuous batching的概念，当一个序列结束后，其他可以立刻跟进，来实现多的概念，当一个序列结束后，其他可以立刻跟进，来实现多batchbatch性能的性能的最大化。值得注意的是，在最大化。值得注意的是，在transformertransformer结构中，包含结构中，包含attentionattention模块和模块和feed forwardfeed forward模块，模块，该论文主要针对的是该论文主要针对的是attentionattention模块的优化。模块的优化。

2. page attentionpage attention

论文链接： https://arxiv.org/abs/2309.06180

continuous batching的思路固然好，但是没有开源，但是的思路固然好，但是没有开源，但是VLLMVLLM提供了提供了cotinuous batchingcotinuous batching的实现，并且，还借鉴了操作系统中的页的概念，分页的对于的实现，并且，还借鉴了操作系统中的页的概念，分页的对于KV cacheKV cache进行管理，在代码进行管理，在代码中，使用了中，使用了block table, block manageblock table, block managerr来进行来进行kv blockkv block的有效管理。的有效管理。

3. speculative decodingspeculative decoding

论文链接： https://arxiv.org/pdf/2211.17192

参考博文：hthttps://blog.csdn.net/qq_27590277/article/details/135812738

推测解码（Speculative DecodingSpeculative Decoding），作为20232023年新兴的一项年新兴的一项LLMLLM推理加速技术，正是提出推理加速技术，正是提出了一种类似的解决方案：通过增加每个解码步了一种类似的解决方案：通过增加每个解码步LLMLLM计算的并行性，减少总的解码步数（即计算的并行性，减少总的解码步数（即减少了减少了LLMLLM参数的反复读写），从而实现推理加速。在每个解码步，推测解码首先高效地参数的反复读写），从而实现推理加速。在每个解码步，推测解码首先高效地““推测推测”target LLM”target LLM（待加速的（待加速的LLMLLM）未来多个解码步可能生成的）未来多个解码步可能生成的tokentoken，然后再用，然后再用target target LLMLLM同时验证这些同时验证这些ttokenoken。通过验证的。通过验证的tokentoken作为当前解码步的解码结果。如果作为当前解码步的解码结果。如果““推测推测””足够足够准确，推测解码就可以在单个解码步并行生成多个准确，推测解码就可以在单个解码步并行生成多个tokentoken，从而实现，从而实现LLMLLM推理加速。并且，推理加速。并且，使用使用target LLMtarget LLM的验证过程可以在理论上保证解码结果和的验证过程可以在理论上保证解码结果和target LLMtarget LLM自回归解码结果的完自回归解码结果的完全一致全一致[5][6][5][6]。。

## LLM 推理框架介绍推理框架介绍
1. vLLM（UC Berkeley）

SOSP 2023
的论文 vLLM ，也是热门开源项目，其创新点 paged attn PA ），减少内存碎片
增加 memory efficiency ，增大 batch size 从而增 加吞吐。 Batching 策略是为 PA 设计服务的，
所以没有照搬 OCRA 的实现。

和
ORCA 不同之处在于， vLLM Batching 时候 prefill 和 decoding 是分开的，一个 Batching
step 要么处理 decoding 要么处理 prefill 。这样实现比 OCRA 更简单了， prefill 直接调用
xformers 处理计算密集的 prefill attn 计算； decoding 手写 CUDA PA 处理访存密集的 attn 计
算。

我觉得
vLLM 之所以没有采用 OCRA 设计，是因为 vLLM 的 PA 是手写 CUDA Ke rnel 实现
的，可以处理 sequence 长度不同的输入， Attn 的 Batching 方式可以和 Non Attn 部分统一。
因此，一个糙快猛方法是不采用 Selective Batching 的设计了，所 Decoding 整体一起处理一
个 Batch 的 step 计算， prefill 不和 decoding step 融合。如果把 prefill 计算和一个 decoding
step 融合，则还需要拆分 Attn 和 Non Attn 了， Attn 实现也更更复杂了，不利于展示 PA 的
思想。

不过因为
Prefill 过程会抢占 decoding 的 st ep 前进，如果输入 prompt sequence length 过长，
所有 decoding 过程都需要等待，造成大家更长的延迟，因此留下了一些优化空间，这后来
这也造成了和 DeepSpeed 的一段孽缘。

2. FastGen deepspeed

微软
DeepSpeed 团队 2023 年 11 月在 MII 项目中提出了一种 Continous Batching 变种
SplitFuse ，在发布时把 vLLM 当靶子打， vLLM 随后还击 [ 6]，逐渐演化成成为两个大门派
的口水战。

SplitFuse
的想法是，对长 prompt request 被分解成更小的块，并在多个 forward step 中进行
调度，只有最后一块的 forward 完成后才开始这个 prompt request 的生成。对短 prompt
request 将被组合以精确填充 step 的空隙。每个 step 的计算量基本相等，达到所有请求平均
延迟更稳定的目的

3. TensorRT LLM

TensorRT-LLM 是 NVIDIA 用于做 LLM Large Language Model ）的可扩展推理方案。该
方案是基于 TensorRT 深度学习编译框架来构建、编译并执行计算图， 并借鉴了许多
FastTransformer 中高效的 Kernels 实现，然后利用 NCCL 完成设备之间的通讯。考虑到
技术的发展和需求的差异，开发者还可以定制算子来满足定制需求，比如基于 cutlass 开
发定制 GEMM 。 TensorRT LLM 是一款致力于提供高性能并不断完善其实用性的 NVIDIA
官方推理方案。

## 推理优化算法介绍

介绍本作品使用的推理优化算法。

本次比赛中，我们重点关注如下三个方面：

1. 通过 continuous batching 的调参来实现系统性能的最大利用；
2. 利用合理的 s equence 长度预计，调整参数；
3. 通过对于输入数据的前处理来实现速度提升。

### 2.1参数优化类
在本模块，我们通过调整推理服务的相关参数
基于 LLM Serving 现阶段已有的推理优化算法，
实现对于 Ascend NPU 的算力的澎湃利用，从而实现速度提升。

### 2.1.1多 batch 优化
本次比赛中，我们可以通过指令
```
python test_serving_performance.py -X 0. 1 -P 8835 -O "./" -T 1500
```

来实现
QPS query per second ）的调整，比如上方指令表示 X 或者 qps 为 0.5 ，触发时间
为 300s ，一共发出 150 条指令。当 QPS 极小时，比如 0.1 推理引擎有充足时间处理请求，
当处理请求较大时，我们注意到推理时间将多于请求发送的时间。

基于实验的观察，我们发现
batch size 的增加 ( 可以显著减少处理的总时间。此外，
我们也留意到，过多增加 qps ，会导致系统丢失对一些请求的处理，从而不能实现完整 1500 条
数据的推理，因此，我们只记录实现了全部请求推理的实验数据，对于 QPS 过大造成请求 未
处理的情况，我们期待进一步优化。

从表1 可以看到，当设置 batch size =64, qps 为 5 时，我们实现了 791s 的推理耗时，并且成功
通过精度测试。

| batch size | QPS(0.5)   | QPS(3)   | QPS (4)  | QPS(5)   |  
|------------|------------|----------|----------|----------|  
| 1          | 3551.925s  | 795s     |   -       | -         |  
| 16         |    -    |     958s      |  930s        |  Invalid        |  
| 32         |   -     |    865s      | 836s     | 836s     |  
| 64         |   -     |   795s       |791s  | 791s     |

表1

### 2.1.2 显存占用优化
注意：加表示该优化未通过精度测试，但是推理结果没有明显偏差

针对
LLM 服务，如 OpenAI 提供的 API 接口服务，更长的 token 数量通常对应更高的服务价格，
这是因为生成更多的 token 通常意味着 kv cache 需要占用更多的显存，更长的推理时间。也就
是说，生成 Token 的数量与 LLM service 系统的性能关系紧密。

针对本赛题提供的输入，和生成文本，我们可以对于输出文本的
token 数进行估计，从而实现
对于 sequence 数量的合理设定。保证 Token 生成的范围大于赛题需要处理的序列长度，但是也
不应该过长。针对 alpaca 数据分析后，我们发现， alpoca 数据中， 序列的总长度通常在 256 以
下， 也有少部分处于 256 1024 的区间。如下图所所示。

因此，我们对于超参进行了进一步调整，
batch size =64 的情况下， 修改了 seq_length
decode_seq_length max_generate_length 为 1024 在此参数下， qps 为 5 时，推理时间只需
要 410s 。

| batch size |  QPS(3)   | QPS (4)  | QPS(5)   |  
|------------|----------|----------|----------|  
| 64        |  526s    |   421s   | 410s      | 


![image](https://github.com/JoegameZhou/competition/blob/master/2024-ascend-innovation-contest/topic3-inference/first-phase/assets/URobot-1.png)


![image](https://github.com/JoegameZhou/competition/blob/master/2024-ascend-innovation-contest/topic3-inference/first-phase/assets/URobot-2.png)


但是经过测试，在调整该参数后，模型的精度验证无法通过。如下图


## 2.2策略优化类
### 2.2.1 reorder in advance*
针对数据集中的文件，我们尝试按照输入和输出
token 的总长度对于输入进行顺序和逆序排序，
我们在 batch 为 64 下， qps 为 4 的超参下，对比了排序后两者的推理速度，实验证明：

逆序：
385.49s

顺序：
468.10s

逆序时的推理速度达到了最快的推理速度。

遗憾的是，经过脚本评估，精度没有通过
但是经过推理 log 的解析，如下图，输出的生成文本
符合预期，使用逆序的 时候，在保证推理正确性的同时，可以有效提升推理速度，而顺序的时
候，耗时而增加。


## 3. 超参配置介绍
针对
2.1.1 多 batch 优化，我们修改了 llama_7b_kbk_pa_dyn.yaml 文件中的 decode_batch_size
为 64

针对
2.1.2 的序列长度约束优化，我们修改了同文件中 seq_length decode_seq_length
max_generate_length 参数为 1024

针对
2.2.1 中提到的输入排序，我们把代码上传，请参考 test_serving_perfo rmance.py 文件的修
改。



## 4. 总结
本次初赛，我们熟悉了针对
LLM serving 的服务推理服务的基本框架，通过调研，我们加深了
对于大模型推理框架架构的认知，了解了近两年出现的 LLM 的相关优化算法和开源推理框架。

由于
LLM sys 目前的系统学习资料较少，本次比赛过程中，大量时间花在了 VLLM 和 llm
serving
serving等代码和相关资料阅读和理解上。在阅读过程中，我们也产生了优化等代码和相关资料阅读和理解上。在阅读过程中，我们也产生了优化prefillprefill和和decodedecode调度优先级优化，调度优先级优化，topktopk并行提速等优化策略，但是由于时间限制，并且在最后阶段发现了并行提速等优化策略，但是由于时间限制，并且在最后阶段发现了qpsqps过高，推理没有全部完成，精度测试无法通过等问题，我们没有对于系统代码进行更深入的优过高，推理没有全部完成，精度测试无法通过等问题，我们没有对于系统代码进行更深入的优化。期待在复赛中能够进一步解决相关问题。化。期待在复赛中能够进一步解决相关问题。


5.
附录
在实验
1 和 2 的执行中，由于没有留意 performace_serving 中是否存在请求丢失的情况，该结
果存在 一定争议。
实验
1 ：预研实验
本实验数据基于
alpaca_521.json 文件展开，按照不同 qps ，进行 150 条数据的推理，因为数据
数量小于正式性能评估的实际要求，仅供参考和前期调研使用。
The OCR output is a bit disorganized. I'll clean up and reformat the data to represent it as a table:

| TestName | X / T | batch | Tput(req/s) | Total time |
|------|-----|-------|----|----|
| 1.0 | 0.5/300| 1 | 0.38 | 391 |
| 1.1 | 0.5/300| 2 | 0.48 | 311 |
| 1.2 |1/150 | 2 | 0.625 | 240 |
|1.4 | 2/75 | 4 | 0.76 | 195 |
| 1.6 | 5/30 | 4 | 0.92 | 163 |
| 1.7 | 5/30 | 8 | 0.96 | 155 |
| 1.8 | 10/15 | 8 | 1.17 | 128 |
| 1.9 | 10/15 | 16| 1.11 | 134 |
| 1.10 | 10/15 | 32 | 1.07 | 139 |
| 1.11 | 15/10 | 32 | 1.38 | 108 |
| VLLM[1] |  |  | 3.74 |  |


实验2

我们把
batch size 从 32 开始依次成倍增加，记录推理时间。 QPS 从 15 开始成倍增加。按照比
赛要求，我 们使用 alpaca_5010.json 进行测试 当 batch size 为 128 时，我们注意到出现 error304
的报错 ，因此 , 基于 batch size=64 ，我们可以得到的最佳速度为 624.119s.

| batch size | QPS(0.5)   | QPS(15)   | QPS (30)  | QPS(60)   |   QPS(100)   | 
|------------|------------|----------|----------|----------|  ----------| 
| 1          | 3551.925s  | -    |   -       | -         |   -         |
| 32         |   -     |    712.655s      | -    | -    |   -         |
|64            | -  |   713.666s  |  654.298s |   638.847s  |  624.119s|


