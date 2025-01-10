# run_llama_infer 实验

    妈的人生是旷野，直接测 mindformers 提供的原生模型的生成速度

----

推理时间与输出长度呈直接的线性关系，输入和输出长度没啥线性关系。

```
Generate speed: ~40 tokens/s 即 0.025s / token
AI core usage: 66%~70%
Total runtime: 2457.07s = 40.95min | 2397.50477s = 39.95min

max(output_lens) = 505                  // <bos> encluded
sum(time) = 2395.26s = 39.921min        // Net infer time
```
