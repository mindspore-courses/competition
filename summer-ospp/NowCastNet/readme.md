基于雷达观测的极端降水短临预测

项目简介

本项目面向灾害预警场景下的极端降水短临预测任务，基于雷达观测数据构建深度学习预测模型，实现未来短时间（如 0–2 小时）的降水强度与空间分布预测。

项目基于 **MindSpore 深度学习框架** 实现 **NowcastNet 网络结构**，并参考 **DGMR（Deep Generative Model for Radar）** 的设计思路，引入判别器模块与对抗训练机制，从而提升预测结果的时空一致性与逼真度。

主要工作包括：

1. **模型搭建：** 使用 MindSpore 实现 NowcastNet 网络的各个核心模块；

2. **对抗训练：** 参考 DGMR 模型设计判别器与训练脚本，完善 NowcastNet 的训练流程；

3. **模型推理：** 使用训练得到的权重文件进行推理与性能评估，完成极端降水短临预测任务。

* * *

项目特点

* 🔧 **基于 MindSpore 实现**，充分利用华为 Ascend 910A 计算平台的算力；

* 🧠 **NowcastNet 模型结构完整复现**，具备编码器、时序动态建模模块与生成式解码器；

* ⚔️ **DGMR 风格判别器设计**，引入时空一致性判别；

* 🌧️ **MRMS 雷达降水数据训练**，数据覆盖广、时空分辨率高；

* ⚡ **分布式训练支持**，在 4 张 Ascend 910A 卡上进行高效并行计算；

* 📈 **多指标评估**，计算 MAE、MSE、RMSE 等预测性能指标。

* * *

模型结构说明

### 1. **NowcastNet 主干网络**

NowcastNet 结合卷积神经网络与时序建模模块，用于学习雷达回波的空间结构与时间演化：

* **编码器（Encoder）：** 提取雷达图像多尺度特征；

* **时序动态模块（Temporal Dynamics）：** 捕捉回波的运动趋势与强度变化；

* **解码器（Decoder）：** 输出未来时间步的降水预测图像。

### 2. **判别器（DGMR风格）**

* 采用多尺度三维卷积结构，对预测序列进行时空判别；

* 通过对抗损失与生成器共同训练；

* 提升生成序列的物理一致性与视觉真实感。

* * *

数据集说明

* **数据来源：** 使用原作者开放的数据集进行训练和推理。
* **产品名称：** MRMS
* **使用产品：** https://cloud.tsinghua.edu.cn/d/b9fb38e5ee7a4dabb2a6/
* **空间范围：** 美国本土
* **用途：** 用于输入过去若干时刻的雷达序列，预测未来降水强度。

* * *

实验环境配置

| 项目               | 配置说明         |
| ---------------- | ------------ |
| **深度学习框架**       | MindSpore    |
| **硬件设备**         | Ascend 910A  |
| **使用卡数**         | 4 张          |
| **数值精度**         | FP32         |
| **数据集**          | MRMS         |
| **系统环境**         | Ubuntu 20.04 |
| **Python 版本**    | ≥ 3.10.0     |
| **MindSpore 版本** | ≥ 2.5.0      |

* * *

项目目录结构
    code_mindspore/
    ├── nowcasting/                     # 核心模块目录
    │   ├── __init__.py
    │   ├── models/                     # 模型定义
    │   │   ├── model_factory.py        # 模型工厂类
    │   │   ├── nowcastnet.py          # NowcastNet主网络
    │   │   └── __init__.py
    │   ├── layers/                     # 网络层实现
    │   │   ├── evolution/              # 演化网络模块
    │   │   │   ├── evolution_network.py
    │   │   │   ├── module.py
    │   │   │   └── __init__.py
    │   │   ├── generation/             # 生成网络模块
    │   │   │   ├── generative_network.py
    │   │   │   ├── discriminators.py   # 判别器
    │   │   │   ├── noise_projector.py  # 噪声投影器
    │   │   │   ├── module.py
    │   │   │   └── __init__.py
    │   │   ├── utils.py               # 工具函数
    │   │   └── __init__.py
    │   ├── data_provider/             # 数据处理模块
    │   │   ├── dataset.py             # 数据集类
    │   │   └── datasets_factory.py    # 数据集工厂
    │   ├── trainer.py                 # 训练器实现
    │   ├── loss.py                    # 损失函数定义
    │   ├── evaluator.py              # 推理执行器
    │   └── evaluations.py            # 评估指标计算
    ├── dataset/                       # 数据集目录
    │   ├── train/                     # 训练数据
    │   ├── valid/                     # 验证数据
    │   └── test/                      # 测试数据
    ├── ckpt/                         # 模型检查点目录
    │   ├── evo/                      # 演化网络权重
    │   └── gen/                      # 生成网络权重
    ├── checkpoints/                  # 训练过程检查点
    ├── train.py                      # 训练脚本
    ├── run.py                       # 推理脚本
    ├── requirements.txt             # 依赖包列表
    └── README.md                    # 项目文档

核心模块说明

* **`nowcasting/models/`**: 包含NowcastNet主网络结构实现
* **`nowcasting/layers/evolution/`**: 演化网络模块，负责预测雷达回波的运动和强度变化
* **`nowcasting/layers/generation/`**: 生成网络模块，包含生成器和判别器的对抗训练
* **`nowcasting/data_provider/`**: 数据加载和预处理模块
* **`nowcasting/trainer.py`**: 分别实现演化和生成两阶段的训练流程
* **`nowcasting/loss.py`**: 定义各种损失函数（演化损失、生成损失、对抗损失等）

* * *

结果分析

模型在 MRMS 测试集上取得如下平均指标：

* **MAE：0.7327 mm/h**

* **MSE：7.4829 mm²/h²**

* **RMSE：2.5074 mm/h**

结果表明 NowcastNet 在 MindSpore 框架下能够较好地捕捉降水的短时空间变化特征，预测性能稳定，能够为极端天气灾害的短临预警提供有效支持。

* * *

🧭 后续工作展望

* 🔬 增大数据量，提升预测精度；

* 🧩 探索 **物理约束损失函数**，提高模型可解释性；

* 🧠 尝试 **基于 Transformer 的时空注意力机制**，提升远期预测能力；

* 🌍 研究模型在不同地区和降水类型下的泛化性能。
