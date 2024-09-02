### （1）微调算法介绍

微调算法基于开源的中英文混合数学运算数据集进行，主要使用的是MindFormers中的Llama3-8b模型。微调的目的是在保持模型原有能力（如SQUAD数据集上的F1 Score和Em Score）不丢失的情况下，提高模型的数学运算准确率。微调后模型需要在测试数据集上生成数学运算结果，确保准确率提升。

- #### 数据集的预处理方式：

  考虑到硬件资源消耗问题，原始数据集为80W条对话数据集，全量参数微调大约需要两天左右的时间。因此我们从原始数据集中随机抽取1/10的数据作为Llama3-8b模型的微调数据，并将数据转换为MindRecord格式。

### （2）超参配置介绍说明

指导手册中提供了对模型进行微调时的超参数配置建议。在微调过程中自行调整这些超参数，以达到最佳效果。微调的超参数涉及训练轮数、学习率、batch_size等，如下所示。

- 增加pet_config配置：

  ```bash
  pet_config:
      pet_type: lora
      lora_rank: 8
      lora_alpha: 16
      lora_dropout: 0.05
      target_modules: '.*wq|.*wv'
  ```

  

- 其他需要修改的参数如下：

  ```
  load_checkpoint: 'path/to/llama3_8b.ckpt' # 填写权重路径
  auto_trans_ckpt: False # 关闭自动权重转换
  use_past: False # 关闭增量推理
  use_parallel: False # 关闭并行模式（单卡）,多卡需要设置True
  only_save_strategy: True
  max_device_memory: "26GB"
  ```

微调超参数具体参考压缩包中的`./yaml/run_llama3_8b_8k_800T_A2_64G_lora_dis_256.yaml`配置文件。

### （3）微调后的权重链接：

https://chi-2024.obs.cn-southwest-2.myhuaweicloud.com/2024-llm/Fine-tuning-lora/new_lora_checkpoint_0.ckpt

### （4）运行环境说明

除了指导手册中的5.2环境配置外，还需要安装MindSpore和MindFormers的指定版本。具体的安装命令如下：

- 安装MindSpore: `pip install mindspore==2.3.0RC2`

- 如果上述命令出错，可使用以下命令下载并安装：

  ```bash
  wget https://2024-ascend-innovation-contest-mindspore.obs.cn-southwest-2.myhuaweicloud.com/topic2-finetune/mindspore-2.3.0rc2-cp39-cp39-linux_aarch64.whl
  pip install mindspore-2.3.0rc2-cp39-cp39-linux_aarch64.whl
  ```

- 安装MindFormers:

  ```bash
  wget https://2024-ascend-innovation-contest-mindspore.obs.cn-southwest-2.myhuaweicloud.com/topic2-finetune/mindformers.zip
  unzip mindformers.zip
  cd mindformers/
  bash build.sh
  ```

此外，还需要配置环境变量，以确保所有路径正确设置与本地文件一致

```bash
export PYTHONPATH="${PYTHONPATH}:/home/ma-user/work/mindformers/"
```



### （5）微调日志、配置文件

微调日志位于`./msrun_log`文件夹下，配置文件位于`./yaml`文件夹下，其中`run_llama3_8b_8k_800T_A2_64G_lora_dis_256.yaml`文件为微调配置文件，`predict_llama3_8b_800T_A2_64G.yaml`为原有能力评估配置文件，`run_llama3_8b_8k_800T_A2_64G_lora_256_eval.yaml`为数学能力推理配置文件。

