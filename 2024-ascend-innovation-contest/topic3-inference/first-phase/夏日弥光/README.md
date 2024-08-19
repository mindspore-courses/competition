### 1. 业界推理优化算法调研

#### 大模型(LLM)推理框架：

- Tensorrt-LLM  Nvidia 半开源
  - 基于 TensorRT 深度学习编译框架来构建、编译并执行计算图，并借鉴了许多 FastTransformer 中高效的 Kernels 实现，然后利用 NCCL 完成设备之间的通讯。
- vLLM:   快速简单易用的大模型推理框架和服务，来自加州大学伯克利分校
  - Python + Cuda kernel 实现
  - 适用于大批量Prompt输入，并对推理速度要求高的场景
- TGI （Text Generation Inference）
  - 依赖HuggingFace模型，并且不需要为核心模型增加多个adapter的场景
- LightLLM:  纯python的推理框架 (Triton kernel)
- **[MLC LLM](https://github.com/mlc-ai/mlc-llm)**：大模型(LLM)高性能通用部署方案，陈天奇(tvm发起者)团队开发.
  - 可在客户端（边缘计算）（例如，在Android或iPhone平台上）本地部署LLM；
- 其他：
  - **[OpenLLM](https://link.zhihu.com/?target=https%3A//github.com/bentoml/OpenLLM)**：为核心模型添加adapter并使用HuggingFace Agents，尤其是不完全依赖PyTorch；
  - **[OpenPPL-LLM](https://github.com/openppl-public/)** ：商汤，实现了LLM 任务全流程深度优化
  - **[CTranslate2](https://link.zhihu.com/?target=https%3A//github.com/OpenNMT/CTranslate2)**：可在CPU上进行推理；
  - **[Ray Serve](https://link.zhihu.com/?target=https%3A//docs.ray.io/en/latest/serve/index.html)**：稳定的Pipeline和灵活的部署，它最适合更成熟的项目；
  - **[DeepSpeed-MII](https://link.zhihu.com/?target=https%3A//github.com/microsoft/DeepSpeed-MII)**：使用DeepSpeed库来部署LLM；

#### 大模型推理常用技术：

**KV Cache**

**PageAttention**

**Continuous Batching**

**Attention with Linear Bias**

### 2.本作品使用的推理优化算法介绍

**KV Cache**

**PageAttention**

**Continuous Batching**

### 3. 超参配置介绍

prefill_batch_size: 8

decode_batch_size: 72

python test_serving_performance.py  -X 2.5  -T 600

### 4. 优化后的推理总时长

794.7744338512421

### 5. 运行环境说明

同教程

### 6. 有关文件

配置文件： llama_7b_kbk_pa_dyn.yaml

推理日志（速度）：test_performance.log

推理日志（精度）：test_logits.log

