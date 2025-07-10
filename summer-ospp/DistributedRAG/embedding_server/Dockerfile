FROM python:3.9-slim

# 在容器内设置一个工作目录
WORKDIR /app

# 关键步骤：在安装Python依赖之前，先安装系统级的编译工具
# apt-get update 更新包列表
# apt-get install -y ... 安装gcc和build-essential
# -y 参数表示自动确认安装
RUN apt-get update && apt-get install -y build-essential gcc

# 复制依赖清单文件到工作目录
COPY requirements.txt .

# 使用国内镜像源安装所有Python依赖
RUN pip install --no-cache-dir -U -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

# 将当前目录下的所有其他文件（主要是app.py）复制到工作目录
COPY . .

# 暴露服务运行的端口（默认为80）
EXPOSE 80

# 容器启动时执行的默认命令：使用uvicorn启动FastAPI应用
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
