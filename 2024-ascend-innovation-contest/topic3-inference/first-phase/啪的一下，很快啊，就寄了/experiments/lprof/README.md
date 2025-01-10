# lprof 性能测试

    只跑模型侧，看 mindformer + mindspore 的性能瓶颈在哪里

----

⚠ lprof 在云端 Ascend 上并不能工作，且只能跑一两个样例，有信号量泄露！！

⚠ 由于设备异构，CPU+PYNATIVE 模式下的性能不一定能代表 NPU+GRAPH 模式下的性能，主要是各算子在不同硬件上实现是不同的；并且配置不同，本地测试 *_debug.yaml 配置中禁用了 recompute 和 past (PageAttn)

```
- Ascend device [aarch64; 192 cores + 192G RAM + 32G HBM]
  - 1 sample: 16.813539743423462
- local CPU [x86_64; i7-11700K + 32G RAM]
  - 1 sample: 8.029037237167358
  - 1+5 samples: 53.53314447402954
```

详细结果见 [run_lprof.py.lprof.txt](./run_lprof.py.lprof.txt)

ℹ 可以差不多得知 Ascend 本质就是个四倍大树莓派 😓
