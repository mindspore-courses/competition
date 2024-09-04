# 大语言模型推理调优报告

## 背景简介及算法调研

大语言模型居高不下的算力消耗是制约其商业服务部署的瓶颈问题。用户访问的不可预测性对大语言模型的推理速度和并发吞吐量提出很高的要求，因此对大语言模型推理速度的优化具有重要商业和学术意义。

目前，绝大多数大语言模型为GPT类网络架构，其生成过程需要对输入的语料token进行循环计算，不断进行k、v、q矩阵乘法计算。然而每次输入token与上次token仅有最后一位不同。因此可以通过缓存上一轮的kv矩阵，下次直接调用的方式来避免重复计算，此方法即为KV Cache算法，本质上是通过增加显存消耗来提升推理速度。此外，KV Cache需要保存每个请求生成过程中所有的缓存，但每次请求生成步数往往不相同，按固定长度分配显存容易造成显存浪费或溢出，因此业界通常使用Page attention方法对推理中的显存进行管理，及对于每个batch分配一个固定大小（blocksize）的显存空间，不够时再继续分配。

## 推理优化算法介绍
以上背景介绍即为不损失精度的情况下，目前最常用稳定性最好的推理调优算法。此算法也内置在了minspore框架内，通过配置文件中if_past可控制是否使用KV Cache，通过page_attention参数控制是否使用page_attention，并可修改相关配置。

本次调优中，首先分析具体时间消耗，如下图所示y轴为时间消耗，x轴为输出日志行数，每次请求均为一个较长时间间隔再加一系列不等的时间间隔。两者占据90%以上的时间。其中较长时间间隔为第一次无KV cache的计算过程，后续较小时间间隔则是增量推理过程。推理过程外的优化空间很小。而对推理本身优化则涉及底层硬件适配。因此，本次选择调优batc_size对速度进行优化。


![image11](https://github.com/JoegameZhou/competition/blob/master/2024-ascend-innovation-contest/topic3-inference/first-phase/assets/Introspection-1.png)


## 超参配置
在原有超参上，将decode_batch_size调整为64，prefill_batch_size调整为32,相应配置文件为llm-serving/configs/llama/llama_7b_kbk_pa_dyn.yaml

```
model_config:
    model_name: 'llama_7b'
    max_generate_length: 4096
    end_token: 2
    seq_length: [4096]
    vocab_size: 32000
    prefill_batch_size: [32]
    decode_batch_size: [64]

```

此时1500条数据的推理时间为821.7s，相关日志为performance_serving/test_batch_64_32_test.log

test.sh设置为python test_serving_performance.py -X 2.5 -P 8835 -O "./" -T 60

精度测试时，由于官方代码限制，配置中需要将decode_batch_size和prefill_batch_size再改回1（与原有配置相同），配置文件为：
llm-serving/configs/llama/llama_7b_kbk_pa_dyn_val.yaml，日志文件为：performance_serving/test_batch_64_32_val.log

注：decode_batch_size和prefill_batch_size大于1时输出精度也经过测试无问题

## 运行说明
运行环境与官方环境相同，所有代码包括日志的压缩包如下：

https://gf-finetune1.obs.cn-southwest-2.myhuaweicloud.com/all_code.zip

可直接在work根目录解压运行，其中checkpoint_download中需要重下权重文件，后续运行命令均与官方相同，测试精度时请用llm-serving/configs/llama/llama_7b_kbk_pa_dyn_val.yaml配置文件










