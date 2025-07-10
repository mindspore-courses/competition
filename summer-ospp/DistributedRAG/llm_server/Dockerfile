# LLM推理可能需要特定环境，但我们先从标准Python镜像开始
# 如果需要GPU支持，后续可以更换为类似 nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 的基础镜像
FROM python:3.9-slim

# 在容器内设置一个工作目录
WORKDIR /app


RUN apt-get update && apt-get install -y build-essential gcc
# 复制依赖清单文件到工作目录
COPY requirements.txt .

# 使用国内镜像源安装所有Python依赖
# 注意：mindspore等大型库的安装可能会比较耗时
RUN pip install  --no-cache-dir -U -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

# 将当前目录下的所有其他文件（主要是app.py）复制到工作目录
COPY . .

# 暴露服务运行的端口（默认为80）
EXPOSE 80

# 容器启动时执行的默认命令：使用uvicorn启动FastAPI应用
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]