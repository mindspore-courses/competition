# 微调算法 & 数据集预处理

微调算法同手册使用lora

对于数据集首先进行了去重, 然后使用规则方法对solution进行了补充, 具体而言将数据集分为几种类别

1. 直接计算, 例子: `{"problem": "计算 7469.98 * -2455.65 等于多少？", "solution": "7469.98 * -2455.65 = -18343656.3870"}`
2. 解方程 (均为相同形式的一元一次方程), 例子: `{"problem": "解方程 -81x + -45 = 0", "solution": "方程的解为：-0.5555555555555556"}`
3. 算均值, 例子: `{"problem": "求以下数据的平均值：[79, 69, 19, 3, 23, 93]", "solution": "平均值为 47.666666666666664"}`
4. 化简分数, 例子: `{"problem": "将分数 5/6 进行简化。", "solution": "最简化的形式为：5/6"}`
5. 计算质量, 例子: `{"problem": "某物体的密度为 10 克/立方厘米，体积为 3 立方厘米，请计算该物体的质量。", "solution": "30 克"}`
6. 算折扣比例, 例子: `  '{"problem": "商品原价为 51 元，打折后的价格为 28 元，请计算打折的折扣比例。", "solution": "45.09803921568628"}`
7. 算面积, 例子: `{"problem": "一个长方形的长为 57 厘米，宽为 84 厘米，请计算其面积。", "solution": "面积为 4788 平方厘米"}`
8. 算平方根, 例子: `{"problem": "计算 6145.00 的平方根？", "solution": "√6145.00 = 78.39005038906404479030928703"}`
9. 算幂 (幂级数为0到5), 例子: `{"problem": "计算 -711.31 的 4 次方？", "solution": "-711.31^4 = 255997460543.58343921"}`
10. 算销售额, 例子: `{"problem": "去年销售额为 32 万元，今年销售额增加了 28%，请计算今年的销售额。", "solution": "40.96"}`
11. 算函数值 (均为次方和乘法), 例子: `{"problem": "当 x = 4.03 时，求函数 y = 57x^54 的值", "solution": "函数的值为：2.769155521409239378157128235E+34"}`
12. 英文

去重后, 中文有538949条, 英文有9993条, 其比例为

{
    "direct computation": 0.654617427706388,
    "eq solve": 0.07301317807710105,
    "area computation": 0.015784909881189635,
    "sqrt computation": 0.07145927985105895,
    "power computation": 0.07142284612946359,
    "function computation": 0.03648472880559331,
    "discount percentage": 0.011431080150544138,
    "mass computation": 0.000182168607976799,
    "mean computation": 0.035903610946147316,
    "sales revenue": 0.011414684975826225,
    "fraction simplify": 8.197587358955955e-05,
    "eg": 0.018204108995121523
}

我们做了以下处理

1. 去除英文数据
2. 针对solution没有列式子直接给出答案的, 我们使用程序补充了列式子的过程
3. 对所有的solution都在前面加上了计算过程:
4. 按比例抽取10w条数据用于训练

这部分转换脚本可以通过下面的命令下载:

```shell
wegt https://nju-hw-finetune.obs.cn-southwest-2.myhuaweicloud.com/caculation_convert.py
```

该文件运行要求原`train.json`文件在同级目录, 并在同级目录创建新文件`train-v2.json`

具体可在代码文件的line 465和line 486修改目录

这里提供一个转换并抽样后的数据文件

```shell
wget https://nju-hw-finetune.obs.cn-southwest-2.myhuaweicloud.com/data-v2/train-v2.json
```

> 由于训练似乎要求数据条数为八的倍数, 所以基于文件转换抽样后可能需要手动删除几条数据

对于该抽样的数据集, 按照手册介绍的将数据转换为`alpaca`格式, 但是做了部分改动, 使数据没有添加prompt, 如下

```json
{
    "id": "1",
    "conversations": [
        {
            "from": "human",
            "value": "计算 967.09 - -688.16 等于多少？"
        },
        {
            "from": "gpt",
            "value": "计算过程：967.09 - -688.16 = 1655.25"
        }
    ]
}
```

修改的`data_converter.py`可以用下面的命令获得

```shell
wget https://nju-hw-finetune.obs.cn-southwest-2.myhuaweicloud.com/data_converter.py
```

在这一步得到的数据可以用下面的命令获得

```shell
wget https://nju-hw-finetune.obs.cn-southwest-2.myhuaweicloud.com/data-v2/train-v2-conversation.json
```

然后按照手册的介绍将数据转化为`MindRecord`数据集, 转化后的数据集可以用下面的命令获得

```shell
wget https://nju-hw-finetune.obs.cn-southwest-2.myhuaweicloud.com/data-v2/train-v2-mindrecord.tar
```



# 超参配置

微调超参在手册的基础上, 对lora超参和学习率进行了修改, 具体而言

```yaml
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.2

lr: 1.e-4
```

配置文件可见与报告同级目录下的文件`run_llama3_8b_8k_800T_A2_64G_lora_dis_256.yaml`

微调参数截图如下, 比例计算为 `3407872 / 8030000000 = 0.00042439252801992525`

![image-20240728173711373](https://raw.githubusercontent.com/bsnmldb/tuchuang/main/img/202407281738753.png)

其余微调日志可在`logs`文件夹中找到

# 权重链接

合并的权重链接可以通过如下命令获得

```shell
wget https://nju-hw-finetune.obs.cn-southwest-2.myhuaweicloud.com/lr4-v2/lr4_v2_3905.ckpt
wget https://nju-hw-finetune.obs.cn-southwest-2.myhuaweicloud.com/lr4-v2/lr4_v2_3124_0.ckpt
wget https://nju-hw-finetune.obs.cn-southwest-2.myhuaweicloud.com/lr4-v2/lr4_v2_2343_0.ckpt
wget https://nju-hw-finetune.obs.cn-southwest-2.myhuaweicloud.com/lr4-v2/lr4_v2_1562_0.ckpt
wget https://nju-hw-finetune.obs.cn-southwest-2.myhuaweicloud.com/lr4-v2/lr4_v2_781_0.ckpt
```

另外, 分布式权重文件在文件夹`https://nju-hw-finetune.obs.cn-southwest-2.myhuaweicloud.com/lr4-v2/lr4_v2_ckpts/`中, 可以按照默认命名 (修改rank`0, 1, 2, 3`的值和step`781, 1562, 2343, 3124, 3905`的值) 的方法下载某个分布式权重文件, 如

```shell
wget https://nju-hw-finetune.obs.cn-southwest-2.myhuaweicloud.com/lr4-v2/lr4_v2_ckpts/rank_0/llama3_8b_rank_0-3905_2.ckpt
```

# 运行环境说明

除手册上提到的, 无额外运行环境配置

# 原有能力评估得分

> mindformers源码包可以通过下面的命令获得
>
> ```shell
> wget https://nju-hw-finetune.obs.cn-southwest-2.myhuaweicloud.com/lr4-v2/mindformers-lr4-v2.zip
> ```

原有能力评估yaml配置文件放在了mindformers文件夹的对应部分 (mindformers/research/llama3)

yaml文件添加了下面一句, 以及将lora的配置修改为和训练时一样

```yaml
min_new_tokens: 1
```

对最后一个ckpt原有能力评估日志文件为`logs/squad_lr4_v2_3905.log`, 评分如下, 相比原有能力 (59.87023775108988, 44.17029511369134) 稍有下降, 但是合格 (53.88321397598089, 39.75326560232221)

```shell
F1 score: 58.4921336310527, Em score: 43.29946782776972, total_count: 2067
```

# 推理方式

数学推理的yaml配置文件也放在了mindformers文件夹的对应部分, 修改与上面相似

另外, 我们的推理输入为裸的jsonl文件, **不添加prompt**, 为了与训练时也没有添加prompt对齐, 自测的分数为0.28以上
