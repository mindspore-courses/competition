import os
import time
import logging
from datetime import datetime
from typing import List

# --- 新增和修改的依赖 ---
import streamlit as st
import ray

# --- 从新文件中导入 Actors 和 Tasks ---
from ray_tasks import EmbeddingActor, LLMActor, parse_and_chunk_document

# --- 配置 ---
RAY_ADDRESS = os.getenv("RAY_ADDRESS", "ray://127.0.0.1:10001") # 从环境变量读取Ray地址
MINIO_HOST = os.getenv("MINIO_HOST", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MILVUS_HOST = os.getenv("MILVUS_HOST", "standalone")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MINIO_BUCKET_NAME = "rag-documents"

# --- 日志配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==============================================================================
# 1. 初始化 Ray 连接
# 在应用启动时，主应用将作为 Ray Client 连接到集群。
# ==============================================================================
try:
    if not ray.is_initialized():
        logging.info(f"正在连接到 Ray 集群: {RAY_ADDRESS}")
        ray.init(address=RAY_ADDRESS, ignore_reinit_error=True)
    logging.info("✅ Ray 连接成功!")
except Exception as e:
    logging.error(f"❌ 无法连接到 Ray 集群: {e}")
    # 在 Streamlit 界面中显示错误，并停止应用
    st.error(f"严重错误：无法连接到 Ray 计算集群，请检查 Ray Head 服务是否正常运行。错误详情: {e}")
    st.stop()


# ==============================================================================
# 2. MilvusClient 类
# ==============================================================================
class MilvusClient:
    def __init__(self, host, port):
        from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection
        self.connections, self.utility, self.Collection = connections, utility, Collection
        self.DataType, self.FieldSchema, self.CollectionSchema = DataType, FieldSchema, CollectionSchema
        for i in range(5): # 减少重试次数，以便更快反馈
            try:
                self.connections.connect("default", host=host, port=port)
                logging.info("✅ Milvus 连接成功。"); return
            except Exception as e:
                logging.warning(f"Milvus 连接尝试 {i+1}/5 失败... Error: {e}"); time.sleep(3)
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

# ==============================================================================
# 3. RAG 核心流程
# ==============================================================================
PROMPT_TEMPLATE = """使用以下上下文来回答用户的问题。如果你不知道答案，请输出“我不知道”。总是使用中文回答。
问题: {question}
可参考的上下文：
···
{context}
···
如果给定的上下文无法让你做出回答，请回答“数据库中没有这个内容，你不知道”。
有用的回答:"""

def execute_rag_pipeline_ray(files_data: List[dict], query: str) -> str:
    logging.info("🚀 ======== 开始执行 Ray RAG 工作流 ========")
    
    # --- 1. 获取 Actor 句柄 ---
    try:
        embedding_actor = ray.get_actor("EmbeddingActor")
        llm_actor = ray.get_actor("LLMActor")
        logging.info("✅ Actor 句柄获取成功。")
    except ValueError:
        # 如果 Actor 不存在
        logging.warning("Actor 未找到，正在创建新的 Actor 实例...")
        embedding_actor = EmbeddingActor.options(name="EmbeddingActor", get_if_exists=True).remote()
        llm_actor = LLMActor.options(name="LLMActor", get_if_exists=True).remote()
        logging.info("✅ 新的 Actor 实例已创建。")

    # --- 2. 并行处理文件 ---
    # 为每个文件创建一个 Ray Task 调用，但不立即执行
    parse_tasks = [parse_and_chunk_document.remote(f['content'], f['name']) for f in files_data]
    logging.info(f"提交了 {len(parse_tasks)} 个文件解析任务到 Ray。")
    
    # --- 3. 向量化用户问题 (可以与文件处理并行) ---
    query_vector_ref = embedding_actor.embed.remote([query])
    logging.info("提交了用户问题向量化任务到 Ray。")

    # --- 4. 等待文件处理完成，并向量化所有文本块 ---
    parsed_results = ray.get(parse_tasks)
    all_chunks = [chunk for result in parsed_results for chunk in result]
    
    if not all_chunks:
        error_msg = "❌ 未能从任何文件中提取文本块。"
        logging.error(error_msg); return error_msg
        
    logging.info(f"所有文件解析完成，共得到 {len(all_chunks)} 个文本块。")
    
    chunk_vectors_ref = embedding_actor.embed.remote(all_chunks)
    logging.info("提交了文本块批量向量化任务到 Ray。")

    # --- 5. 初始化外部客户端 (Milvus, MinIO) ---
    # 这部分不是计算密集型的，可以在主进程中执行
    milvus_client = MilvusClient(host=MILVUS_HOST, port=MILVUS_PORT)

    # --- 6. 等待向量化结果，并存入 Milvus ---
    # 同时等待问题向量和文本块向量的结果，最大化并行
    # [query_vector], [chunk_vectors] = ray.get([query_vector_ref, chunk_vectors_ref])
    query_vector, chunk_vectors = ray.get([query_vector_ref, chunk_vectors_ref])
    
    if not chunk_vectors or not query_vector:
        error_msg = "❌ 向量化失败，请检查 EmbeddingActor 的日志。"; 
        logging.error(error_msg); return error_msg

    logging.info("✅ 问题和文本块向量化全部完成。") # 移动到了检查之后
        
    collection_name = f"rag_session_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    milvus_client.insert(collection_name, all_chunks, chunk_vectors)
    logging.info(f"已将 {len(all_chunks)} 个向量存入 Milvus 集合 '{collection_name}'。")

    # --- 7. 检索 ---
    retrieved_docs = milvus_client.search(collection_name, query_vector)
    context = "\n---\n".join(retrieved_docs)
    logging.info("✅ 从 Milvus 检索到相关上下文。")

    # --- 8. 生成回答 ---
    prompt = PROMPT_TEMPLATE.format(question=query, context=context)
    logging.info(f"生成回答的 prompt:\n{prompt}\n")
    answer_ref = llm_actor.generate.remote(prompt)
    logging.info("提交了最终答案生成任务到 Ray。")
    
    final_response = ray.get(answer_ref)
    logging.info("✅ 获得最终回答。")
    
    logging.info("🏁 ======== Ray RAG 工作流执行完毕 ========")
    return final_response


# ==============================================================================
# 4. Streamlit 界面封装
# ==============================================================================
def run_streamlit_app():
    """
    主函数，用于渲染Streamlit界面并处理用户交互。
    """
    # 页面基础配置
    st.set_page_config(page_title="分布式RAG应用 (Ray版)", layout="wide")
    st.title("🚀 分布式RAG应用 (Ray 统一计算后端)")
    st.markdown("上传文件并提问，系统将通过 Ray 分布式后端并行处理数据并生成回答。")

    # 初始化会话状态
    if "response" not in st.session_state:
        st.session_state.response = "请在下方提交问题和文件，我会在这里给出回答..."

    # --- 界面布局 ---
    with st.form("rag_form"):
        query = st.text_input(
            "请输入你的问题:",
            placeholder="例如：这张截图里显示的核心数据是什么？"
        )
        
        # 更新文件上传组件以接受图片文件
        uploaded_files = st.file_uploader(
            "上传文件（支持图片、PDF、Markdown、文本），可多选: 🖼️",
            accept_multiple_files=True,
            type=['png', 'jpg', 'jpeg', 'md', 'txt', 'pdf']
        )
        
        submit_button = st.form_submit_button("提交")

    # --- 逻辑处理 ---
    if submit_button and query and uploaded_files:
        with st.spinner("系统正在通过 Ray 分布式后端处理中，请稍候..."):
            try:
                files_data = [{'name': f.name, 'content': f.getvalue()} for f in uploaded_files]
                logging.info(f"Streamlit 接收到查询: '{query}' 和 {len(files_data)} 个文件。")
                final_response = execute_rag_pipeline_ray(files_data=files_data, query=query)
                st.session_state.response = final_response
            except Exception as e:
                st.session_state.response = f"处理过程中发生严重错误: {e}"
                logging.error(f"Streamlit UI层捕获到异常: {e}", exc_info=True)
    elif submit_button:
        st.error("错误：请确保您已输入问题并上传了文件。")

    # --- 显示回答区域 ---
    st.subheader("模型的回答:")
    st.text_area("回答内容", value=st.session_state.response, height=400, disabled=True, label_visibility="collapsed")

if __name__ == "__main__":
    run_streamlit_app()