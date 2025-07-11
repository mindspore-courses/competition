
FROM python:3.9-slim

# 在容器内设置一个工作目录
WORKDIR /app


RUN apt-get update && apt-get install -y build-essential gcc
# 复制依赖清单文件到工作目录
COPY requirements.txt .

RUN pip install  --no-cache-dir -U -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

COPY . .

EXPOSE 80

# 容器启动时执行的默认命令：使用uvicorn启动FastAPI应用
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]