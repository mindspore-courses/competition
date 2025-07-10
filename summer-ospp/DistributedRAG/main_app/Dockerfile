# 使用一个标准的、轻量级的Python 3.9镜像作为基础
FROM python:3.9-slim

# 设置容器内的时区为亚洲/上海，避免日志时间戳问题
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 在容器内设置一个工作目录
WORKDIR /app

# 复制依赖清单文件到工作目录
# 先复制并安装依赖，可以有效利用Docker的层缓存机制
COPY requirements.txt .

# 使用国内镜像源安装所有Python依赖，加速构建过程
# --no-cache-dir 减小镜像体积
RUN pip install --no-cache-dir -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

# 将当前目录下的所有其他文件（主要是你的应用代码，如 main_app/ 目录）复制到工作目录
COPY . .

# 暴露Streamlit应用的默认端口
# Gradio是7860, Streamlit默认是8501。我们这里沿用之前建议的7860端口
EXPOSE 7860

# 容器启动时执行的默认命令
# 使用 streamlit run 来启动应用
# --server.port 指定端口
# --server.address 0.0.0.0 允许从容器外部访问
# --server.enableCORS=false 是一个好习惯，除非你确实需要跨域
CMD ["streamlit", "run", "main.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.enableCORS=false"]
