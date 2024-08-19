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

比赛排名第一


