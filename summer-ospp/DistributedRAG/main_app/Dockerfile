# 使用一个标准的 Python 镜像作为基础
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app


RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    libsm6 libxrender1 libxext6 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 复制整个 main_app 目录到容器中
COPY . /app/

# 安装前端和客户端所需的轻量级依赖
# 注意：这里不再需要 mindspore, mindnlp 等重型库
RUN pip install --no-cache-dir \
    "streamlit" \
    "marshmallow~=3.20.0" \
    "ray==2.9.3" \
    "pymilvus==2.4.1" \
    "boto3" \
    "botocore" \
    #"mindspore~=2.3.0"
    "mindspore~=2.6.0" \
    "mindnlp~=0.4.0" \
    "sentencepiece" \
    "tiktoken~=0.8.0" \
    "PyPDF2~=3.0.1" \
    "markdown~=3.7" \
    "beautifulsoup4~=4.12.3"\
    "rapidocr-onnxruntime" \
    "Pillow" \
    "jinja2" -i https://mirrors.aliyun.com/pypi/simple/

# 容器暴露的端口
EXPOSE 7860

# 启动 Streamlit 
CMD ["streamlit", "run", "main.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.enableCORS=false"]