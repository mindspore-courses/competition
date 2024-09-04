# 作品报告

参考文献：Goat: Fine-tuned LLaMA Outperforms GPT-4 on Arithmetic Tasks

如果保留测试数据和训练集分布一致，在保留5位有效数字的情况下，应该取得50%左右准确率合理。

微调权重获取obs地址：'obs://mindformers-finetune/new_lora_checkpoint_0.ckpt'

执行代码流程参考1.10

额外依赖包：pip install pandas

训练耗时：4卡5 epoch, 9到10小时左右，样本4万+，主要相比baseline， seq_length=768

代码obs路径：obs://mindformers-finetune/code.zip

微调分割数据路径：obs://mindformers-finetune/train-data.json

## 一.微调方案

### 1.1和baseline一致基于LORA进行微调，微调参数如下：
```
pet_config:
      pet_type: lora
      # configuration of lora
      lora_rank: 64
      lora_alpha: 64
      lora_dropout: 0.05
      target_modules: '.*wq|.*wk|.*wv|.*wo'

```

### 1.2微调seq_len 由原来的256修改为768

seq_length: 768

### 1.3微调参数比例

Network Parameters: 54525952

54525952 / 8030000000 = 0.679%

### 1.4 微调样本量及训练时间

4.2万+样本，4卡batch_size=16（单卡batch_size = 4）配置下，微调5个epoch花费9小时+

如果需要保证原有能力多次采样数据，多次训练都能达标，只训练一个epoch即可

训练采样数据obs路径（全部来源于train.json采样）:

obs://mindformers-finetune/train-data.json

obs://mindformers-finetune/train-data-conversation.json

### 1.5.原有能力保持

F1 score: 66.843193143867, Em score: 48.08901790033865, total_count: 2067

日志文件：eval_squad.log

配置文件：
run_llama3_8b_8k_800T_A2_64G_lora_256_base_eval.yaml

修改：min_new_tokens: 1,不增加该配置，测试原有能力大多是空字符串结果
。


### 1.6权重文件obs路径
obs://mindformers-finetune/new_lora_checkpoint_0.ckpt

### 1.7微调日志文件
msrun_log.zip

### 1.8推理流程修改及数字结果提取说明

预测参考脚本：predict.sh

评估提取数字参考脚本：eval_val.py

run_llama3_test.py修改如下：

只在输入problem末尾追加了"\n\n### Response:"
```
 # 加载json格式推理数据
    predict_data = []
    prompt_no_input ="{problem}\n\n### Response:"
    with open(input_dir, 'r', encoding='utf-8') as file:
        # print(file)
        for line in file:
            line = json.loads(line)
            # print(line['problem'])
            pro_list = prompt_no_input.format_map(line)
            predict_data.append(pro_list)

```
结果提取输出字符串中最后一个数字类型的匹配项，
包括分数：r'-?\d+\/\d+’    比如：1/3
浮点数/整数：r’-?\d+\.?\d*' 比如：108, 1.08896等
科学计数法：暂时忽略

### 1.9 额外安装包

`pip install pandas `

### 1.10 代码执行说明

安装环境：下载llama3-8B.ckpt和tokenizer.model并安装包

`bash init_run.sh `

添加环境变量：添加mindformers包环境变量

`source name.sh `

处理数据：下载train.json 然后处理-》train-data.json-》train-data-conversation.json-》train-fastchat768.mindrecord

`bash init_data.sh`

目前配置40960条训练数据,单卡batch_size=4

开始微调：

多卡配置文件：`run_llama3_8b_8k_800T_A2_64G_lora_dis_768.yaml`

如果用多卡执行：`bash train_four.sh `

然后合并权重：`bash merge_checkpoint.sh `

如果单卡执行：

单卡配置文件：run_llama3_8b_8k_800T_A2_64G_lora_dis_256.yaml

`bash train_single.sh`


验证原有能力：
先从obs加载训练好的权重到本地(参考ob_data.ipynb中代码)
```
import moxing as mox
mox.file.copy('obs://mindformers-finetune/new_lora_checkpoint_0.ckpt', '/home/ma-user/work/mindformers/research/output/checkpoint/rank_0/new_lora_checkpoint_0.ckpt')
```
然后执行：`bash eval_squad.sh`


## 二.数据预处理
模板修改为如下：
```
    "prompt_no_input": (
        "{problem}\n\n### Response:")
```

只在problem后面添加\n\n### Response:

所有answer能应用思维链CoT的应用CoT, 不能应用的提取其中的数值为answer，移除其它内容。

对应脚本为：
data_process.py 得到train-data.json

data_converter.py 得到 train-data-conversation.json


## 三.COT 方案
方案效果提升主要基于大模型思维链 CoT（Chain of Thought）增加推理的中间步骤，从而使得难以学习的任务变得容易学习
目前处理的任务类型有：乘法、除法（包括一元一次方程等），求数列的均值类型

### 3.1乘法
包括的模板类型有如下正则表达式范畴：

re.search(r'计算\s?-?\d+\.?\d*\s?\*\s?-?\d+\.?\d*', row['problem']) is not None

re.search(r'一个长方形的长为 \d+ 厘米，宽为 \d+ 厘米，请计算其面积', row['problem']) is not None

re.search(r'某物体的密度为 \d+ 克/立方厘米，体积为 \d+ 立方厘米，请计算该物体的质量', row['problem']) is not None

re.search(r'去年销售额为 \d+ 万元，今年销售额增加了 \d+%，请计算今年的销售额', row['problem']) is not None

乘法CoT方案如下：

结果示例：'2650.13 * 6314.58 = 2650.13 * (6000 + 300 + 10 + 4 + 0.5 + 0.08) = 2650.13 * 6000 + 2650.13 * 300 + 2650.13 * 10 + 2650.13 * 4 + 2650.13 * 0.5 + 2650.13 * 0.08 = 15900780.00 + 795039.00 + 26501.30 + 10600.52 + 1325.065 + 212.0104 = 16695819.00 + 26501.30 + 10600.52 + 1325.065 + 212.0104 = 16722320.30 + 10600.52 + 1325.065 + 212.0104 = 16732920.82 + 1325.065 + 212.0104 = 16734245.885 + 212.0104 = 16734457.8954\nfinal, -2650.13 * -6314.58 = 16734457.8954'

乘法当其中存在乘数有效位数为1位（非0的数字位数）的情况下模型容易学习（2位需要大量数据，3位以上难以学习），因此方案将其中一位乘数进行分解后得到一个数和一位有效数字的数相乘的和，最后求和。推导过程中忽略乘数正负号，最后总结的时候，得到合理的数字正负号：\nfinal, -2650.13 * -6314.58 = 16734457.8954


### 3.2 除法
包括的模板类型有如下正则表达式范畴：

re.search(r'计算\s?-?\d+\.?\d*\s?\/\s?-?\d+\.?\d*', row['problem']) is not None

re.search(r'解方程 -?\d+x \+ -?\d+ = 0', row['problem']) is not None

re.search(r'商品原价为 \d+ 元，打折后的价格为 \d+ 元，请计算打折的折扣比例', row['problem']) is not None

除法CoT方案：

结果示例：

2292.89 / -7520.36
'2292890000.00 - 7520.36 * 300000 = 2292890000.00 - 2256108000.00 = 36782000.00\n36782000.00 - 7520.36 * 4000 = 36782000.00 - 30081440.00 = 6700560.00\n6700560.00 - 7520.36 * 800 = 6700560.00 - 6016288.00 = 684272.00\n684272.00 - 7520.36 * 90 = 684272.00 - 676832.40 = 7439.60\nTherefore, 2292890000.00 / 7520.36 = 304890 R 7439.60\nfinal, 2292.89 / -7520.36 ~= -0.30489'

先确定结果需要保留的小数点位数k,方案中取值5,也就是小数点后保留5位。就先将被除数乘以10**(k+1),这样最后得到整数结果部分和余数。最后总结学习根据最后一位四舍五入，并移位得到小数解。


### 3.3求数列均值
正则表达式如下：

re.search(r'求以下数据的平均值：\[.+\]', row['problem']) is not None

求均值CoT方案：

结果示例：

[78, 84, 8, 62, 56, 31, 25, 22, 93]
'78 + 84 + 8 + 62 + 56 + 31 + 25 + 22 + 93 = 162 + 8 + 62 + 56 + 31 + 25 + 22 + 93 = 170 + 62 + 56 + 31 + 25 + 22 + 93 = 232 + 56 + 31 + 25 + 22 + 93 = 288 + 31 + 25 + 22 + 93 = 319 + 25 + 22 + 93 = 344 + 22 + 93 = 366 + 93 = 459\n计算 459 / 9\n459000000 - 9 * 50000000 = 459000000 - 450000000 = 9000000\n9000000 - 9 * 1000000 = 9000000 - 9000000 = 0\nTherefore, 459000000 / 9 = 51000000\nfinal, 459 / 9 ~= 51'

先推导数列求和CoT,然后追加除法CoT

### 3.4.目前学习的样本范畴如下（涵盖了80%+的样本呢）

加法：re.search(r'计算\s?-?\d+\.?\d*\s?\+\s?-?\d+\.?\d*', row['problem']) is not None 

减法：re.search(r'计算\s?-?\d+\.?\d*\s?\-\s?-?\d+\.?\d*', row['problem']) is not None 

乘法：re.search(r'计算\s?-?\d+\.?\d*\s?\*\s?-?\d+\.?\d*', row['problem']) is not None 

除法： re.search(r'计算\s?-?\d+\.?\d*\s?\/\s?-?\d+\.?\d*', row['problem']) is not None 

一元一次方程：re.search(r'解方程 -?\d+x \+ -?\d+ = 0', row['problem']) is not None 

求面积：re.search(r'一个长方形的长为 \d+ 厘米，宽为 \d+ 厘米，请计算其面积', row['problem']) is not None \
 求质量：re.search(r'某物体的密度为 \d+ 克/立方厘米，体积为 \d+ 立方厘米，请计算该物体的质量', row['problem']) is not None \

 计算折扣：re.search(r'商品原价为 \d+ 元，打折后的价格为 \d+ 元，请计算打折的折扣比例', row['problem']) is not None \

计算销售额：re.search(r'去年销售额为 \d+ 万元，今年销售额增加了 \d+%，请计算今年的销售额', row['problem']) is not None \

求数列均值：re.search(r'求以下数据的平均值：\[.+\]', row['problem']) is not None 

简单分数简化：re.search(r'将分数 \d+/\d+ 进行简化', row['problem']) is not None

N次方中的0和1次方：re.search(r'计算\s?-?\d+.?\d*\s?的\s?\d+\s?次方?', row['problem'])
2次方转换成乘法，3次方，4次方转换成两次乘法推导待完成


