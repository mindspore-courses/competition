# McpMultiTravelPlanner - 基于MCP协议整合多源信息实现多模态动态旅游规划系统

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![MindSpore](https://img.shields.io/badge/MindSpore-2.0+-green.svg)](https://www.mindspore.cn/)

## 📖 项目简介

McpMultiTravelPlanner 基于模型上下文协议（MCP）整合用户偏好、实时交通/天气数据、景点热度、消费成本等多源信息，并结合 SOTA模型实现个性化行程生成、动态路线优化及体验模拟。

### 核心组件

1. **前端界面**：基于 Vue3 的 Web 应用
2. **强化学习**：基于 ppo 算法考虑距离、花费、景点评分优化旅游方案
3. **后端接口**：基于 flask 实现旅游方案生成、对话等接口

## 🚀 快速开始

### 环境要求

- **python**：3.10
- **vue**：3.5.21
- **vite**：7.1.2
- **image**：mindspore_2.4.10-cann_8.0.0-py_3.10-euler_2.10.11-aarch64-snt9

### 安装步骤

1. **克隆项目**
```bash
git clone https://github.com/mindspore-courses/competition.git
cd summer-ospp/McpMultiTravelPlanner
```

2. **准备模型文件**
```bash
pip install huggingface-hub
huggingface-cli login

cd llm_service/app/models
huggingface-cli download \
  Qwen/Qwen2-7B-Instruct \
  --local-dir ./Qwen2-7B-Instruct \
  --local-dir-use-symlinks False
```

3. **配置环境**
```bash
cd ../../
pip install -r requestment.txt

cd ../TravelVue
npm install
```

4. **MCP工具配置**
- 获取指定地点景点列表：利用高德地图的地点检索API， 5000次/月
- 获取指定景点附近的酒店：利用百度地图的地点检索API，100次/天
- 获取小红书上相关的旅游帖子作为参考：在coze中利用小红书插件搭建工作流并使用API
- 获取两地之间的步行/骑行/驾车/公共交通路线：利用百度地图的路径规划API，30000次/天
- 获取指定地区的天气情况：利用百度地图天气API，300000次/天
- 获取指定地点的经纬度（用于路径规划）：利用百度地图地理编码API，5000次/天
- 获取指定地点附近的美食：利用百度地图地点检索服务，通过美食相关关键字进行检索，100次/天

其中，百度地图和高德地图均需要在对应的开发者平台进行注册申请并替换相应的密钥。

获取小红书上相关的旅游帖子在coze中搭建的工作流样式为
![img.png](imgs/img.png)

5. **启动服务**
```bash
npm run dev

cd ../llm_service
python run.py
```

6. **访问应用**
- 打开浏览器访问：`http://localhost:5173/`


## 🔧 开发指南

### 项目结构

```
─McpMultiTravelPlanner
│  ├─imgs                   # 测试结果等相关图片
│  ├─llm_service            # 后端应用
│  │  │  requirements.txt   # Python依赖包列表
│  │  │  run.py             # 应用启动入口
│  │  └─app                 # 应用主目录
│  │      │  config.py      # 应用配置
│  │      │  routes.py      # 接口路由
│  │      ├─models          # 存放大语言模型
│  │      ├─ppo
│  │      │      flexible_tourist_model.zip     # 训练好的PPO模型权重
│  │      │      get_ans.py                     # 获取PPO结果
│  │      │      touristAttractions.py          # PPO算法
│  │      └─utils                               # 工具
│  │              McpTool.py                    # MCP工具
│  │              sample_tool.py                # 常用工具
│  │              weather_district_id.xlsx      # 天气API地区ID映射表
│  ├─ppo
│  │      test.py                    # 算法测试
│  │      touristAttractions.py      # PPO算法
│  │      train.py                   # 算法训练
│  └─TravelVue              # 前端VUE应用
│      ├─public             # 静态资源
│      └─src                # 源代码
│          ├─api            # 后端API调用封装
│          ├─assets         # 样式表
│          ├─components     # Vue组件
│          ├─router         # 路由配置
│          ├─utils          # 工具函数
│          └─views          # 页面组件
```

## 🙏 致谢

- [MindSpore](https://www.mindspore.cn/) - 深度学习框架
- [Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct) - 大语言模型
- [Vue3](https://huggingface.co/Qwen/Qwen2-7B-Instruct) - Web 应用框架
- [Flask](https://github.com/pallets/flask) - 后端应用框架
