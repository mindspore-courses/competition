# DistributedRAG - 分布式检索增强生成系统

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![MindSpore](https://img.shields.io/badge/MindSpore-2.0+-green.svg)](https://www.mindspore.cn/)
[![Ray](https://img.shields.io/badge/Ray-2.0+-orange.svg)](https://ray.io/)

## 📖 项目简介

DistributedRAG 是一个基于 Ray 分布式计算框架和 MindSpore 深度学习框架构建的高性能检索增强生成（RAG）系统。该系统支持多模态文档处理、智能检索、重排序和生成，能够处理文本、PDF、图片、音频等多种格式的文档，并提供联网搜索功能。

### 核心组件

1. **前端界面**：基于 Streamlit 的 Web 应用
2. **分布式计算**：Ray 集群管理任务调度和资源分配
3. **向量存储**：Milvus 向量数据库存储文档嵌入
4. **对象存储**：MinIO 存储原始文档和模型文件
5. **元数据管理**：ETCD 管理集群状态和配置

## 🚀 快速开始

### 环境要求

- **Docker**：20.10+
- **Docker Compose**：2.0+

### 安装步骤

1. **克隆项目**
```bash
git clone https://github.com/mindspore-courses/competition.git
cd DistributedRAG
```

2. **准备模型文件**
```bash
mkdir -p rag_models_cache
```

3. **启动服务**

**mindnlp 版本：**
```bash
docker-compose -f docker-compose1.yml up -d
```

**mindspore 原生推理-支持CPU/GPU**
```bash
docker-compose -f docker-compose2.yml up -d
```

4. **访问应用**
- 打开浏览器访问：`http://localhost:7860`
- Ray Dashboard：`http://localhost:8265`
- MinIO 控制台：`http://localhost:9001`

### 配置说明

#### 环境变量

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `RAY_ADDRESS` | `ray://127.0.0.1:10001` | Ray 集群地址 |
| `MILVUS_HOST` | `standalone` | Milvus 主机地址 |
| `MILVUS_PORT` | `19530` | Milvus 端口 |
| `MINIO_HOST` | `minio:9000` | MinIO 主机地址 |
| `MINIO_ACCESS_KEY` | `minioadmin` | MinIO 访问密钥 |
| `MINIO_SECRET_KEY` | `minioadmin` | MinIO 秘密密钥 |

#### 模型配置

demo内提供的两套模型配置：

1. **Set1**：
   - 使用 MindNLP 框架
   - 支持 BAAI/bge-base-zh-v1.5 嵌入模型
   - 支持 MiniCPM-2B 语言模型

2. **Set2**：
   - 使用原生 MindSpore
   - 支持 Qwen3-Embedding 嵌入模型
   - 支持 Qwen2.5-1.5B-Instruct 语言模型
   - 支持 Qwen3-Reranker 重排序模型

## 🔧 开发指南

### 项目结构

```
DistributedRAG/
├── main_app1/                 # CPU 版本应用
│   ├── main.py               # 主应用入口
│   ├── ray_tasks.py          # Ray 任务定义
│   └── test.py               # 测试脚本
├── main_app2/                 # GPU 版本应用
│   ├── main.py               # 主应用入口
│   ├── ray_tasks.py          # Ray 任务定义
│   ├── qwen_embedding_model.py    # 嵌入模型
│   ├── qwen_reranker_model.py     # 重排序模型
│   └── qwen_causal_lm.py          # 语言模型
├── Dockerfiles/              # Docker 配置文件
│   ├── Dockerfile.ray_set1   # CPU 版本 Ray 镜像
│   ├── Dockerfile.ray_set2   # GPU 版本 Ray 镜像
│   ├── Dockerfile.set1       # CPU 版本应用镜像
│   └── Dockerfile.set2       # GPU 版本应用镜像
├── docker-compose1.yml       # CPU 版本编排文件
├── docker-compose2.yml       # GPU 版本编排文件
├── rag_models_cache/         # 模型缓存目录
└── volumes/                  # 数据持久化目录
    ├── etcd/                 # ETCD 数据
    ├── milvus/               # Milvus 数据
    └── minio/                # MinIO 数据
```

## 🙏 致谢

- [Ray](https://ray.io/) - 分布式计算框架
- [MindSpore](https://www.mindspore.cn/) - 深度学习框架
- [Qwen](https://github.com/QwenLM/Qwen) - 大语言模型
- [Milvus](https://milvus.io/) - 向量数据库
- [Streamlit](https://streamlit.io/) - Web 应用框架
