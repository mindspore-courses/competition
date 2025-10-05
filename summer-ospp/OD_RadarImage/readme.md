# 任务名称
  基于MindSpore的4D毫米波雷达-图像双视角融合的3D检测模型复现

# 背景描述
  自动驾驶感知系统需要兼顾性能、鲁棒性和成本，而现有的摄像头和激光雷达方案在恶劣天气或复杂场景下仍存在不足，4D毫米波雷达与摄像头融合可提升复杂天气下的3D检测鲁棒性。DPFT提出双视角投影（前视+鸟瞰）和跨模态注意力机制，在K-Radar数据集上实现了SOTA性能。目前该方案仅支持PyTorch框架，将其迁移至MindSpore将推动国产自动驾驶技术的发展，并为多模态感知提供新的解决方案。

# 数据集
  将KRadar数据集中的前摄像头数据、4D雷达张量、传感器标定数据、标签安装如下目录组织
  ```
  └── 1
      ├── cam-front
      ├── info_calib
      ├── info_label
      └── radar_tesseract
  ```
  关于pypcd库的修改：
      将pypcd/pypcd.py第15行的 import cStringIO as sio 替换为 import io as sio
  
  # 模型权重
  提供adapted_v1.ckpt和adapted_v2.ckpt两个版本的模型权重，分别基于Kradar数据集的revision v1.0 和 v2.0训练。

  # 预处理

    ```python
      python prepare.py --src ./dataset_dir --cfg ./config/kradar.json --dst ./processed_data
    ```

  # 模型评估

    ```python
      python evaluate.py --src ./dataset_dir --cfg ./config/kradar.json --dst ./processed_data
    ```
  # 可视化Demo
    ```python
      python demo.py 
    ```