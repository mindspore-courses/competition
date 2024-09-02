# 推理优化赛道报告提交


## 本作品使用的推理优化算法介绍

使用的是官方提供的推理优化算法

超参配置介绍：-X  1 -T 1500

测试时间推理之前需要预热一下，代码如下：

```
cd /home/ma-user/work/llm-serving/
curl 127.0.0.1:8835/models/llama2/generate \
-X POST \
-d '{"inputs":" I love Beijing, because","parameters":{"max_new_tokens":56, "do_sample":"True", "return_full_text":"True"}, "stream":"True"}' \
-H 'Content-Type: application/json'

```
   
优化后的推理总时长：2102 s
   
运行环境为官方提供环境，无额外环境依赖
   
llm-serving和performance_serving源码包链接：

https://ms-tuili.obs.cn-southwest-2.myhuaweicloud.com/llmserving.zip

https://ms-tuili.obs.cn-southwest-2.myhuaweicloud.com/performance.zip
