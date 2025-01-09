# mixtral


## 优化点：

ppt中基础优化

修改moe模块:通过stack将所有expert拼接并并行forward，再根据expert mask以及对应的权重对输出进行并行后处理

对不必要的dropout操作进行判断并过滤

自定义one_hot算子
# qwen2_moe

## 优化点：（与mixtral一致）

