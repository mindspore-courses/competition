优化方法：
本次优化代码都在transformers/models/t5/modeling_t5.py
1.LayerNorm算子替换
2.INT64转INT32
将所有TENSOR索引替换为OPS的函数