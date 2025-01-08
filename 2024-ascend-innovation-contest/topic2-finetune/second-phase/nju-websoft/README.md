# 数据集

对官方提供的数据集进行去重, 官方提供的数据集有 27604 条数据, 去重后有 27562 条数据

去重的脚本和去重后用于训练的数据如下

+  [preprocess.py](preprocess.py) 
+  [mmlu_train_unique.json](mmlu_train_unique.json) 

然后和手册一样的处理，转化为 `mindrecord` 格式

# 微调算法

使用 lora 算法, 学习率 1e-4, epoch 10, dropout 0.1, rank 8, target_modules .*wq|.\*wk|.\*wv|.\*wo|.\*w1|.\*w2|.\*w3

其他和原配置相同, 微调的配置文件为  [finetune_internlm_7b_lora_mmlu_64G.yaml](finetune_internlm_7b_lora_mmlu_64G.yaml) 

# 原有能力评估

使用最后一步权重 (step 8610), 原有能力评估分数为 `F1 score: 55.51201548096568, Em score: 34.88147073052733`, 评估日志文件和配置文件为 [eval_squad.log](eval_squad.log) 、[predict_internlm_7b_eval_squad.yaml](predict_internlm_7b_eval_squad.yaml)  

# 推理评估

使用最后一步权重 (step 8610), 自测分数为 0.9985, 评估配置文件为 [predict_internlm_7b_mmlu.yaml](predict_internlm_7b_mmlu.yaml) 



# 参数占比

参数占比如图, 为 19988480 / 7321000000 = 0.00273029367572736

![image-20241030135646631](https://raw.githubusercontent.com/bsnmldb/tuchuang/main/img/202410301356670.png)
