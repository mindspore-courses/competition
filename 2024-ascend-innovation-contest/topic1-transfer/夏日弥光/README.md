### Bert

#### 1. 测试结果

512 tokens:   6.697ms

![性能测试结果](.\Bert\性能测试结果.png)

11 tokens:  5.67ms

#### 2.优化方式

- 原始：48.037 ms 

- 绑核：41.604 ms 
- fuse qkv  linear : 37.83  ms
- fuse transpose：28 ~ 36 ms
- 局部静态化： 22~ 24ms 
- mint 算子替换:  21ms
- fix dtype: 19ms
- 整图静态化: 5~6 ms



### Clip

#### 1. 测试结果

10.118ms

![性能测试结果](.\Clip\性能测试结果.png)



#### 2. 优化方式

- 原始：  90ms
- 绑核：  76 ms
- 局部静态化：43ms
- fuse qkv Linear and   transpose： 34 ~ 40ms
- FlashAttention  34 ~ 40ms
- mint算子替换   27ms
- 整图静态化 10ms