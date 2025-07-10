# main_app/main.py

import gradio as gr
import requests
import os
import PyPDF2
import markdown
from bs4 import BeautifulSoup
import re
import tiktoken
import time
import logging
from datetime import datetime
from typing import List, Dict
from tempfile import NamedTemporaryFile

# ------------------- 服务连接配置 (与之前相同) -------------------
EMBEDDING_SERVER_URL = os.getenv("EMBEDDING_SERVER_URL", "http://embedding-server/embed")
LLM_SERVER_URL = os.getenv("LLM_SERVER_URL", "http://llm-server/generate")
MINIO_HOST = os.getenv("MINIO_HOST", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MILVUS_HOST = os.getenv("MILVUS_HOST", "standalone")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MINIO_BUCKET_NAME = "rag-documents"

# --- 核心逻辑类 (与之前相同) ---
enc = tiktoken.get_encoding("cl100k_base")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FileProcessor:
    # ... (此处省略与之前版本完全相同的FileProcessor代码) ...
    @staticmethod
    def read_pdf(file_stream) -> str:
        reader = PyPDF2.PdfReader(file_stream)
        text = "".join(page.extract_text() for page in reader.pages)
        return text
    @staticmethod
    def read_markdown(file_stream) -> str:
        md_text = file_stream.read().decode('utf-8')
        html_text = markdown.markdown(md_text)
        soup = BeautifulSoup(html_text, 'html.parser')
        plain_text = soup.get_text()
        return re.sub(r'http\S+', '', plain_text)
    @staticmethod
    def read_text(file_stream) -> str:
        return file_stream.read().decode('utf-8')
    @classmethod
    def read_file_content(cls, file_stream, file_name: str) -> str:
        if file_name.endswith('.pdf'): return cls.read_pdf(file_stream)
        elif file_name.endswith('.md'): return cls.read_markdown(file_stream)
        elif file_name.endswith('.txt'): return cls.read_text(file_stream)
        else: logging.warning(f"不支持的文件类型: {file_name}"); return ""
    @staticmethod
    def chunk_text(text: str, max_token_len: int = 600, cover_content: int = 150) -> List[str]:
        chunks = []
        tokens = enc.encode(text)
        i = 0
        while i < len(tokens):
            end = min(i + max_token_len, len(tokens))
            chunk_tokens = tokens[i:end]
            chunk_text = enc.decode(chunk_tokens)
            chunks.append(chunk_text)
            i += (max_token_len - cover_content)
        return chunks

class ServiceClient:
    # ... (此处省略与之前版本完全相同的ServiceClient代码) ...
    @staticmethod
    def get_embeddings(texts: List[str]) -> List[List[float]]:
        try:
            response = requests.post(EMBEDDING_SERVER_URL, json={"texts": texts}, timeout=60)
            response.raise_for_status()
            return response.json()["embeddings"]
        except requests.exceptions.RequestException as e:
            logging.error(f"调用Embedding服务失败: {e}"); return []
    @staticmethod
    def generate_response(prompt: str) -> str:
        try:
            response = requests.post(LLM_SERVER_URL, json={"prompt": prompt}, timeout=None)
            response.raise_for_status()
            return response.json()["response"]
        except requests.exceptions.RequestException as e:
            logging.error(f"调用LLM服务失败: {e}"); return f"错误：无法连接到LLM服务。 {e}"

class MilvusClient:
    # ... (此处省略与之前版本完全相同的MilvusClient代码) ...
    def __init__(self, host, port):
        from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection
        self.connections, self.utility, self.Collection = connections, utility, Collection
        self.DataType, self.FieldSchema, self.CollectionSchema = DataType, FieldSchema, CollectionSchema
        for i in range(20):
            try:
                self.connections.connect("default", host=host, port=port)
                logging.info("✅ Milvus连接成功。"); return
            except Exception as e:
                logging.warning(f"Milvus连接尝试 {i+1}/20 失败，正在重试... Error: {e}"); time.sleep(5)
        raise ConnectionError("错误：多次尝试后无法连接到Milvus。")
    def create_or_get_collection(self, collection_name: str, dim: int = 768) -> 'Collection':
        if self.utility.has_collection(collection_name): return self.Collection(collection_name)
        fields = [ self.FieldSchema(name="pk", dtype=self.DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100),
                   self.FieldSchema(name="text", dtype=self.DataType.VARCHAR, max_length=65535),
                   self.FieldSchema(name="embedding", dtype=self.DataType.FLOAT_VECTOR, dim=dim) ]
        schema = self.CollectionSchema(fields, "RAG知识库集合")
        collection = self.Collection(name=collection_name, schema=schema)
        index_params = {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 1024}}
        collection.create_index(field_name="embedding", index_params=index_params)
        return collection
    def insert(self, collection_name: str, texts: List[str], vectors: List[List[float]]):
        collection = self.create_or_get_collection(collection_name)
        collection.insert([texts, vectors]); collection.flush()
    def search(self, collection_name: str, query_vector: List[List[float]], top_k: int = 3) -> List[str]:
        if not self.utility.has_collection(collection_name): return ["错误：知识库集合不存在。"]
        collection = self.Collection(collection_name); collection.load()
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = collection.search(data=query_vector, anns_field="embedding", param=search_params, limit=top_k, output_fields=["text"])
        return [hit.entity.get('text') for hit in results[0]] if results else []


# --- RAG核心流程编排 (重构后) ---
# 我们将之前的测试流程封装成一个更通用的业务逻辑函数

PROMPT_TEMPLATE = """使用以下上下文来回答用户的问题。如果你不知道答案，请输出“我不知道”。总是使用中文回答。
问题: {question}
可参考的上下文：
···
{context}
···
如果给定的上下文无法让你做出回答，请回答“数据库中没有这个内容，你不知道”。
有用的回答:"""

def execute_rag_pipeline(files: List[str], query: str) -> str:
    """
    这是一个封装了完整RAG工作流的核心业务函数。
    它接收文件路径列表和问题，返回最终的答案。
    """
    logging.info("=============================================")
    logging.info("          开始执行RAG工作流          ")
    logging.info("=============================================")

    # 1. 初始化客户端
    milvus_client = MilvusClient(host=MILVUS_HOST, port=MILVUS_PORT)
    # ... MinIO客户端初始化 ...
    from botocore.exceptions import ClientError
    import boto3
    try:
        minio_client = boto3.client('s3', endpoint_url=f'http://{MINIO_HOST}', aws_access_key_id=MINIO_ACCESS_KEY, aws_secret_access_key=MINIO_SECRET_KEY)
        try:
            minio_client.head_bucket(Bucket=MINIO_BUCKET_NAME)
        except ClientError as e:
            if e.response['Error']['Code'] == '404': minio_client.create_bucket(Bucket=MINIO_BUCKET_NAME)
            else: raise
    except Exception as e:
        error_msg = f"❌ MinIO客户端初始化失败: {e}"; logging.error(error_msg); return error_msg

    # 2. 文件处理、向量化并存入Milvus
    collection_name = f"rag_session_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    all_chunks = []
    for file_path in files:
        file_name = os.path.basename(file_path)
        try:
            # 上传到MinIO
            with open(file_path, "rb") as f: minio_client.upload_fileobj(f, MINIO_BUCKET_NAME, file_name)
            # 从MinIO下载并处理
            response = minio_client.get_object(Bucket=MINIO_BUCKET_NAME, Key=file_name)
            content = FileProcessor.read_file_content(response['Body'], file_name)
            chunks = FileProcessor.chunk_text(content)
            all_chunks.extend(chunks)
        except Exception as e:
            error_msg = f"处理文件 '{file_name}' 时出错: {e}"; logging.error(error_msg); return error_msg
    
    if not all_chunks: error_msg = "❌ 未能从任何文件中提取文本块。"; logging.error(error_msg); return error_msg

    chunk_vectors = ServiceClient.get_embeddings(all_chunks)
    if not chunk_vectors: error_msg = "❌ 向量化失败，请检查Embedding服务。"; logging.error(error_msg); return error_msg
    
    milvus_client.insert(collection_name, all_chunks, chunk_vectors)

    # 3. 检索
    query_vector = ServiceClient.get_embeddings([query])
    if not query_vector: error_msg = "❌ 用户问题向量化失败。"; logging.error(error_msg); return error_msg
    
    retrieved_docs = milvus_client.search(collection_name, query_vector)

    # 4. 生成回答
    context = "\n---\n".join(retrieved_docs)
    prompt = PROMPT_TEMPLATE.format(question=query, context=context)
    final_response = ServiceClient.generate_response(prompt)
    
    logging.info("=============================================")
    logging.info("              RAG工作流执行完毕              ")
    logging.info("=============================================")
    
    return final_response


# --- Gradio 界面封装 ---

def gradio_interface_function(query: str, files: List[NamedTemporaryFile]) -> str:
    """
    这是专门为Gradio界面编写的封装函数。
    它负责将Gradio的输入格式转换为我们核心业务函数所需的格式。
    """
    if not query:
        return "错误：请输入您的问题。"
    if not files:
        return "错误：请上传至少一个文件。"

    # Gradio上传的文件是临时文件对象，我们需要获取它们的实际路径
    # .name 属性存储了临时文件在磁盘上的路径
    file_paths = [file.name for file in files]
    logging.info(f"Gradio接收到查询: '{query}' 和 {len(file_paths)} 个文件。")
    
    # 调用我们已经验证过的核心RAG工作流函数
    return execute_rag_pipeline(files=file_paths, query=query)

# 创建Gradio界面
interface = gr.Interface(
    fn=gradio_interface_function, # 将Gradio与我们的封装函数绑定
    inputs=[
        gr.Textbox(label="请输入你的问题"),
        gr.Files(label="上传文件（支持 .md, .txt, .pdf）", file_count="multiple")
    ],
    outputs=gr.Textbox(label="RAG 应用的回答", lines=10, interactive=False),
    title="分布式RAG应用",
    description="上传文件并提问，系统将基于文件内容，通过分布式的Embedding和LLM服务生成回答。",
    allow_flagging="never"
)


# --- 主程序入口 ---
if __name__ == "__main__":
    # 移除之前的测试脚本逻辑，替换为启动Gradio Web服务
    logging.info("正在启动Gradio Web UI...")
    # 在Docker容器内，必须将服务器绑定到0.0.0.0才能从外部访问
    interface.launch(server_name="0.0.0.0", server_port=7860)

