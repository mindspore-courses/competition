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

# ------------------- 服务连接配置 (与之前相同) -------------------
EMBEDDING_SERVER_URL = os.getenv("EMBEDDING_SERVER_URL", "http://embedding-server/embed")
LLM_SERVER_URL = os.getenv("LLM_SERVER_URL", "http://llm-server/generate")
MINIO_HOST = os.getenv("MINIO_HOST", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MILVUS_HOST = os.getenv("MILVUS_HOST", "standalone") # 确保这里是正确的服务名
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MINIO_BUCKET_NAME = "rag-documents"

# --- 核心逻辑类 (FileProcessor, ServiceClient, MilvusClient - 与之前相同) ---
# 初始化tiktoken编码器
enc = tiktoken.get_encoding("cl100k_base")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FileProcessor:
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
        if file_name.endswith('.pdf'):
            return cls.read_pdf(file_stream)
        elif file_name.endswith('.md'):
            return cls.read_markdown(file_stream)
        elif file_name.endswith('.txt'):
            return cls.read_text(file_stream)
        else:
            logging.warning(f"不支持的文件类型: {file_name}")
            return ""

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
    @staticmethod
    def get_embeddings(texts: List[str]) -> List[List[float]]:
        try:
            response = requests.post(EMBEDDING_SERVER_URL, json={"texts": texts}, timeout=60)
            response.raise_for_status()
            return response.json()["embeddings"]
        except requests.exceptions.RequestException as e:
            logging.error(f"调用Embedding服务失败: {e}")
            return []

    @staticmethod
    def generate_response(prompt: str) -> str:
        try:
            # 移除超时限制，让程序一直等待LLM输出结果
            response = requests.post(LLM_SERVER_URL, json={"prompt": prompt}, timeout=None) #120
            response.raise_for_status()
            return response.json()["response"]
        except requests.exceptions.RequestException as e:
            logging.error(f"调用LLM服务失败: {e}")
            return f"错误：无法连接到LLM服务。 {e}"

class MilvusClient:
    def __init__(self, host, port):
        from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection
        self.connections = connections
        self.utility = utility
        self.Collection = Collection
        self.DataType = DataType
        self.FieldSchema = FieldSchema
        self.CollectionSchema = CollectionSchema
        
        for i in range(20): # 增加重试次数
            try:
                self.connections.connect("default", host=host, port=port)
                logging.info("✅ Milvus连接成功。")
                return
            except Exception as e:
                logging.warning(f"Milvus连接尝试 {i+1}/20 失败，正在重试... Error: {e}")
                time.sleep(5) # 增加等待时间
        raise ConnectionError("错误：多次尝试后无法连接到Milvus。")

    def create_or_get_collection(self, collection_name: str, dim: int = 768) -> 'Collection':
        if self.utility.has_collection(collection_name):
            logging.info(f"集合 '{collection_name}' 已存在。")
            return self.Collection(collection_name)
        
        logging.info(f"集合 '{collection_name}' 不存在，正在创建...")
        fields = [
            self.FieldSchema(name="pk", dtype=self.DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100),
            self.FieldSchema(name="text", dtype=self.DataType.VARCHAR, max_length=65535),
            self.FieldSchema(name="embedding", dtype=self.DataType.FLOAT_VECTOR, dim=dim)
        ]
        schema = self.CollectionSchema(fields, "RAG知识库集合")
        collection = self.Collection(name=collection_name, schema=schema)
        
        index_params = {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 1024}}
        collection.create_index(field_name="embedding", index_params=index_params)
        logging.info("✅ 集合创建并索引成功。")
        return collection

    def insert(self, collection_name: str, texts: List[str], vectors: List[List[float]]):
        collection = self.create_or_get_collection(collection_name)
        entities = [texts, vectors]
        collection.insert(entities)
        collection.flush()
        logging.info(f"成功向 '{collection_name}' 插入 {len(texts)} 条数据。")

    def search(self, collection_name: str, query_vector: List[List[float]], top_k: int = 3) -> List[str]:
        if not self.utility.has_collection(collection_name):
            return ["错误：知识库集合不存在。"]
            
        collection = self.Collection(collection_name)
        collection.load()
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = collection.search(
            data=query_vector,
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["text"]
        )
        return [hit.entity.get('text') for hit in results[0]] if results else []

# --- RAG核心流程编排 ---

PROMPT_TEMPLATE = """使用以下上下文来回答用户的问题。如果你不知道答案，请输出“我不知道”。总是使用中文回答。
问题: {question}
可参考的上下文：
···
{context}
···
如果给定的上下文无法让你做出回答，请回答“数据库中没有这个内容，你不知道”。
有用的回答:"""

def run_rag_workflow(test_files: List[str], query: str):
    """
    这是一个自动化的RAG工作流测试函数。
    """
    logging.info("=============================================")
    logging.info("          开始执行RAG工作流测试          ")
    logging.info("=============================================")

    # 1. 初始化所有服务的客户端
    logging.info("步骤 1: 初始化所有服务的客户端...")
    milvus_client = MilvusClient(host=MILVUS_HOST, port=MILVUS_PORT)
    
    # --- MinIO客户端初始化逻辑修正 ---
    from botocore.exceptions import ClientError
    import boto3
    try:
        minio_client = boto3.client('s3', endpoint_url=f'http://{MINIO_HOST}',
                                    aws_access_key_id=MINIO_ACCESS_KEY,
                                    aws_secret_access_key=MINIO_SECRET_KEY)
        
        # 检查存储桶是否存在，如果不存在（即捕获到404错误），则创建它
        try:
            minio_client.head_bucket(Bucket=MINIO_BUCKET_NAME)
            logging.info(f"✅ MinIO存储桶 '{MINIO_BUCKET_NAME}' 已存在。")
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                logging.info(f"MinIO存储桶 '{MINIO_BUCKET_NAME}' 不存在，正在创建...")
                minio_client.create_bucket(Bucket=MINIO_BUCKET_NAME)
                logging.info(f"✅ MinIO存储桶 '{MINIO_BUCKET_NAME}' 创建成功。")
            else:
                # 如果是其他错误，则重新抛出
                raise
        logging.info("✅ MinIO客户端初始化成功。")
    except (ClientError, Exception) as e:
        logging.error(f"❌ MinIO客户端初始化失败: {e}")
        return
    # --- 修正结束 ---

    # 2. 上传文件到MinIO并进行处理
    logging.info(f"\n步骤 2: 处理并上传 {len(test_files)} 个文件到MinIO...")
    all_chunks = []
    for file_path in test_files:
        file_name = os.path.basename(file_path)
        try:
            with open(file_path, "rb") as f:
                minio_client.upload_fileobj(f, MINIO_BUCKET_NAME, file_name)
            logging.info(f"  - 文件 '{file_name}' 已上传到MinIO。")

            response = minio_client.get_object(Bucket=MINIO_BUCKET_NAME, Key=file_name)
            content = FileProcessor.read_file_content(response['Body'], file_name)
            chunks = FileProcessor.chunk_text(content)
            all_chunks.extend(chunks)
            logging.info(f"  - 文件 '{file_name}' 处理完成，生成 {len(chunks)} 个文本块。")
        except Exception as e:
            logging.error(f"处理文件 '{file_name}' 时出错: {e}")
            continue
    
    if not all_chunks:
        logging.error("❌ 未能从任何文件中提取文本块，测试终止。")
        return

    # 3. 向量化并存入Milvus
    logging.info(f"\n步骤 3: 向量化 {len(all_chunks)} 个文本块并存入Milvus...")
    collection_name = f"test_run_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    chunk_vectors = ServiceClient.get_embeddings(all_chunks)
    if not chunk_vectors:
        logging.error("❌ 向量化失败，请检查Embedding服务。测试终止。")
        return
    milvus_client.insert(collection_name, all_chunks, chunk_vectors)
    logging.info("✅ 数据已成功存入Milvus。")

    # 4. 检索
    logging.info("\n步骤 4: 检索与用户问题相关的知识...")
    logging.info(f"  - 用户问题: '{query}'")
    query_vector = ServiceClient.get_embeddings([query])
    if not query_vector:
        logging.error("❌ 用户问题向量化失败。测试终止。")
        return
    retrieved_docs = milvus_client.search(collection_name, query_vector)
    logging.info(f"✅ 检索完成，找到 {len(retrieved_docs)} 个相关文档片段。")
    for i, doc in enumerate(retrieved_docs):
        logging.info(f"  - 相关片段 {i+1}: '{doc[:100]}...'")

    # 5. 生成回答
    logging.info("\n步骤 5: 构建Prompt并调用LLM生成最终回答...")
    context = "\n---\n".join(retrieved_docs)
    prompt = PROMPT_TEMPLATE.format(question=query, context=context)
    logging.info(f"  - 构建的Prompt (部分): '{prompt[:200]}...'")
    final_response = ServiceClient.generate_response(prompt)

    logging.info("\n=============================================")
    logging.info("              RAG工作流测试完成              ")
    logging.info("=============================================")
    logging.info(f"最终生成的回答:\n{final_response}")

# --- 主程序入口 ---
if __name__ == "__main__":
    TEST_DATA_DIR = "/app/test_data"
    TEST_QUERY = "请问MindSpore是什么？"

    if not os.path.exists(TEST_DATA_DIR) or not os.listdir(TEST_DATA_DIR):
        logging.error("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        logging.error(f"错误: 测试数据目录 '{TEST_DATA_DIR}' 不存在或为空。")
        logging.error("请在您的项目根目录下创建一个 'test_data' 文件夹，并放入至少一个测试文件。")
        logging.error("并确保 docker-compose.yml 文件中 main_app 服务的 volumes 部分包含：")
        logging.error("  - ./test_data:/app/test_data")
        logging.error("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    else:
        # 增加一个启动前的等待时间，确保所有服务都已就绪
        logging.info("等待15秒，确保所有服务（特别是MinIO和Milvus）完全启动...")
        time.sleep(15)
        test_files = [os.path.join(TEST_DATA_DIR, f) for f in os.listdir(TEST_DATA_DIR)]
        run_rag_workflow(test_files, TEST_QUERY)

