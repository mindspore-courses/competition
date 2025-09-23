FROM python:3.9-slim

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

RUN pip install --no-cache-dir \
    "streamlit" \
    "marshmallow~=3.20.0" \
    "ray==2.9.3" \
    "pymilvus==2.4.1" \
    "boto3" \
    "botocore" \
    "mindspore~=2.6.0" \
    "mindnlp~=0.4.0" \
    "sentencepiece" \
    "tiktoken~=0.8.0" \
    "PyPDF2~=3.0.1" \
    "markdown~=3.7" \
    "beautifulsoup4~=4.12.3"\
    "rapidocr-onnxruntime" \
    "Pillow" \
    "langchain-text-splitters" \
    "jinja2" \
    "python-docx" \         
    "python-pptx" \         
    "pandas" \              
    "openai-whisper" \      
    "soundfile" \   
    "duckduckgo-search" \ 
    "requests" \       
    -i https://mirrors.aliyun.com/pypi/simple/

# 容器暴露的端口
EXPOSE 7860

# 启动 Streamlit 应用
CMD ["streamlit", "run", "main.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.enableCORS=false"]