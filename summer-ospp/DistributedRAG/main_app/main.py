# main_app/main.py

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

# ------------------- Streamlit相关依赖 -------------------
import streamlit as st
import tempfile # 用于处理Streamlit上传的文件

EMBEDDING_SERVER_URL = os.getenv("EMBEDDING_SERVER_URL", "http://embedding-server/embed")
LLM_SERVER_URL = os.getenv("LLM_SERVER_URL", "http://llm-server/generate")
MINIO_HOST = os.getenv("MINIO_HOST", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MILVUS_HOST = os.getenv("MILVUS_HOST", "standalone")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MINIO_BUCKET_NAME = "rag-documents"

enc = tiktoken.get_encoding("cl100k_base")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FileProcessor:
    """
    文件处理工具类，支持PDF、Markdown、TXT等格式的读取与分块。
    """
    @staticmethod
    def read_pdf(file_stream) -> str:
        """
        读取PDF文件内容，返回纯文本。
        """
        reader = PyPDF2.PdfReader(file_stream)
        text = "".join(page.extract_text() for page in reader.pages)
        return text
    @staticmethod
    def read_markdown(file_stream) -> str:
        """
        读取Markdown文件内容，转为纯文本。
        """
        md_text = file_stream.read().decode('utf-8')
        html_text = markdown.markdown(md_text)
        soup = BeautifulSoup(html_text, 'html.parser')
        plain_text = soup.get_text()
        return re.sub(r'http\S+', '', plain_text)
    @staticmethod
    def read_text(file_stream) -> str:
        """
        读取TXT文件内容，返回字符串。
        """
        return file_stream.read().decode('utf-8')
    @classmethod
    def read_file_content(cls, file_stream, file_name: str) -> str:
        """
        根据文件扩展名自动选择读取方式。
        """
        if file_name.endswith('.pdf'): return cls.read_pdf(file_stream)
        elif file_name.endswith('.md'): return cls.read_markdown(file_stream)
        elif file_name.endswith('.txt'): return cls.read_text(file_stream)
        else: logging.warning(f"不支持的文件类型: {file_name}"); return ""
    @staticmethod
    def chunk_text(text: str, max_token_len: int = 600, cover_content: int = 150) -> List[str]:
        """
        将长文本按最大token数分块，支持重叠。
        """
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
    """
    服务调用工具类，负责与Embedding和LLM服务交互。
    """
    @staticmethod
    def get_embeddings(texts: List[str]) -> List[List[float]]:
        """
        调用Embedding服务，将文本批量转为向量。
        """
        try:
            response = requests.post(EMBEDDING_SERVER_URL, json={"texts": texts}, timeout=60)
            response.raise_for_status()
            return response.json()["embeddings"]
        except requests.exceptions.RequestException as e:
            logging.error(f"调用Embedding服务失败: {e}"); return []
    @staticmethod
    def generate_response(prompt: str) -> str:
        """
        调用LLM服务，生成问题的回答。
        """
        try:
            response = requests.post(LLM_SERVER_URL, json={"prompt": prompt}, timeout=None)
            response.raise_for_status()
            return response.json()["response"]
        except requests.exceptions.RequestException as e:
            logging.error(f"调用LLM服务失败: {e}"); return f"错误：无法连接到LLM服务。 {e}"

class MilvusClient:
    """
    Milvus 数据库操作工具类，支持集合创建、插入、检索。
    """
    def __init__(self, host, port):
        """
        初始化Milvus连接。
        """
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
        """
        创建或获取指定名称的Milvus集合。
        """
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
        """
        向指定集合插入文本及其向量。
        """
        collection = self.create_or_get_collection(collection_name)
        collection.insert([texts, vectors]); collection.flush()
    def search(self, collection_name: str, query_vector: List[List[float]], top_k: int = 3) -> List[str]:
        """
        检索与查询向量最相似的文本。
        """
        if not self.utility.has_collection(collection_name): return ["错误：知识库集合不存在。"]
        collection = self.Collection(collection_name); collection.load()
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = collection.search(data=query_vector, anns_field="embedding", param=search_params, limit=top_k, output_fields=["text"])
        return [hit.entity.get('text') for hit in results[0]] if results else []


# --- RAG核心流程编排 (与之前相同，无需改动) ---
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
    RAG核心流程：文件上传、分块、向量化、入库、检索、生成回答。
    :param files: 文件路径列表
    :param query: 用户问题
    :return: 最终生成的回答
    """
    logging.info("=============================================")
    logging.info("          开始执行RAG工作流          ")
    logging.info("=============================================")
    # 1. 初始化客户端
    milvus_client = MilvusClient(host=MILVUS_HOST, port=MILVUS_PORT)
    from botocore.exceptions import ClientError
    import boto3
    try:
        # 初始化MinIO客户端
        minio_client = boto3.client('s3', endpoint_url=f'http://{MINIO_HOST}', aws_access_key_id=MINIO_ACCESS_KEY, aws_secret_access_key=MINIO_SECRET_KEY)
        try:
            # 检查桶是否存在，不存在则创建
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
            # 上传文件到MinIO
            with open(file_path, "rb") as f: minio_client.upload_fileobj(f, MINIO_BUCKET_NAME, file_name)
            # 从MinIO下载并处理文件内容
            response = minio_client.get_object(Bucket=MINIO_BUCKET_NAME, Key=file_name)
            content = FileProcessor.read_file_content(response['Body'], file_name)
            # 文本分块
            chunks = FileProcessor.chunk_text(content)
            all_chunks.extend(chunks)
        except Exception as e:
            error_msg = f"处理文件 '{file_name}' 时出错: {e}"; logging.error(error_msg); return error_msg
    if not all_chunks: error_msg = "❌ 未能从任何文件中提取文本块。"; logging.error(error_msg); return error_msg
    # 文本向量化
    chunk_vectors = ServiceClient.get_embeddings(all_chunks)
    if not chunk_vectors: error_msg = "❌ 向量化失败，请检查Embedding服务。"; logging.error(error_msg); return error_msg
    # 入库
    milvus_client.insert(collection_name, all_chunks, chunk_vectors)
    # 3. 检索
    query_vector = ServiceClient.get_embeddings([query])
    if not query_vector: error_msg = "❌ 用户问题向量化失败。"; logging.error(error_msg); return error_msg
    retrieved_docs = milvus_client.search(collection_name, query_vector)
    # 4. 生成回答
    context = "\n---\n".join(retrieved_docs)
    prompt = PROMPT_TEMPLATE.format(question=query, context=context)
    logging.info(prompt)
    final_response = ServiceClient.generate_response(prompt)
    logging.info("=============================================")
    logging.info("              RAG工作流执行完毕              ")
    logging.info("=============================================")
    return final_response


# --- Streamlit 界面封装 ---

def run_streamlit_app():
    """
    主函数，用于渲染Streamlit界面并处理用户交互。
    """
    # 页面基础配置
    st.set_page_config(page_title="分布式RAG应用", layout="wide")
    st.title("🚀 分布式RAG应用")
    st.markdown("上传文件并提问，系统将基于文件内容，通过分布式的Embedding和LLM服务生成回答。")

    # 初始化会话状态，用于保存回答
    if "response" not in st.session_state:
        st.session_state.response = "请在下方提交问题和文件，我会在这里给出回答..."

    # --- 界面布局 ---
    with st.form("rag_form"):
        query = st.text_input(
            "请输入你的问题:",
            placeholder="例如：这篇报告的核心结论是什么？"
        )
        uploaded_files = st.file_uploader(
            "上传文件（支持 .md, .txt, .pdf），可多选:",
            accept_multiple_files=True,
            type=['md', 'txt', 'pdf']
        )
        submit_button = st.form_submit_button("提交问题和文件")

    # --- 逻辑处理 ---
    if submit_button:
        if not query:
            st.error("错误：请输入您的问题。")
        elif not uploaded_files:
            st.error("错误：请上传至少一个文件。")
        else:
            temp_file_paths = []
            try:
                with st.spinner("系统正在处理中，请稍候..."):
                    for uploaded_file in uploaded_files:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            temp_file_paths.append(tmp_file.name)
                    
                    logging.info(f"Streamlit 接收到查询: '{query}' 和 {len(temp_file_paths)} 个文件。")
                    
                    final_response = execute_rag_pipeline(files=temp_file_paths, query=query)
                    
                    # 更新会话状态中的回答
                    st.session_state.response = final_response

            except Exception as e:
                st.session_state.response = f"处理过程中发生严重错误: {e}"
                logging.error(f"Streamlit UI层捕获到异常: {e}")
            finally:
                # 无论成功与否，都要清理临时文件
                for path in temp_file_paths:
                    if os.path.exists(path):
                        os.remove(path)
                        logging.info(f"已清理临时文件: {path}")

    # --- 显示回答区域 ---
    st.subheader("模型的回答:")
    st.text_area("回答内容", value=st.session_state.response, height=400, disabled=True, label_visibility="collapsed")


# --- 主程序入口 ---
if __name__ == "__main__":
    # 启动Streamlit Web服务
    logging.info("正在启动 Streamlit Web UI...")
    run_streamlit_app()