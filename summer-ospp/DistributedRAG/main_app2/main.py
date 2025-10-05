import os
import time
import logging
import re
from datetime import datetime
from typing import List, Dict, Tuple

import streamlit as st
import ray
from ddgs import DDGS
from bs4 import BeautifulSoup
import requests

from qwen_embedding_model import QwenEmbeddingModel
from qwen_reranker_model import QwenRerankerModel
from qwen_causal_lm import QwenCausalLM
from ray_tasks import parse_and_chunk_document

RAY_ADDRESS = os.getenv("RAY_ADDRESS", "ray://127.0.0.1:10001")
MILVUS_HOST = os.getenv("MILVUS_HOST", "standalone")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MAX_OPTIMIZATION_ATTEMPTS = 2

EMBEDDING_MODEL_PATH = "/app/.mindnlp/model/Qwen3-Embedding"
RERANKER_MODEL_PATH = "/app/.mindnlp/model/Qwen3-Reranker"
LLM_MODEL_PATH = "/app/.mindnlp/model/Qwen2_5-1_5B-Instruct"
EMBEDDING_DIM = 1024

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    if not ray.is_initialized():
        logging.info(f"正在连接到 Ray 集群: {RAY_ADDRESS}")
        ray.init(address=RAY_ADDRESS, ignore_reinit_error=True)
    logging.info("✅ Ray 连接成功!")
except Exception as e:
    logging.error(f"❌ 无法连接到 Ray 集群: {e}")
    st.error(f"严重错误：无法连接到 Ray 计算集群。错误详情: {e}")
    st.stop()

class EmbeddingActor:
    def __init__(self):
        self.model = QwenEmbeddingModel(EMBEDDING_MODEL_PATH)
    def embed(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts)
        return embeddings.tolist()

class RerankerActor:
    def __init__(self):
        self.model = QwenRerankerModel(RERANKER_MODEL_PATH)
    def compute_score(self, sentence_pairs: List[Tuple[str, str]]) -> List[float]:
        return self.model.compute_score(sentence_pairs)

class LLMActor:
    def __init__(self):
        self.model = QwenCausalLM(LLM_MODEL_PATH)
    def generate(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        return self.model.generate(messages)

class MilvusClient:
    def __init__(self, host, port):
        from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection
        self.connections, self.utility, self.Collection = connections, utility, Collection
        self.DataType, self.FieldSchema, self.CollectionSchema = DataType, FieldSchema, CollectionSchema
        for i in range(5):
            try:
                self.connections.connect("default", host=host, port=port)
                logging.info("✅ Milvus 连接成功。")
                return
            except Exception as e:
                logging.warning(f"Milvus 连接尝试 {i+1}/5 失败... Error: {e}")
                time.sleep(3)
        raise ConnectionError("错误：多次尝试后无法连接到Milvus。")

    def create_or_get_collection(self, collection_name: str, dim: int = EMBEDDING_DIM) -> 'Collection':
        if self.utility.has_collection(collection_name):
            return self.Collection(collection_name)
        fields = [
            self.FieldSchema(name="pk", dtype=self.DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100),
            self.FieldSchema(name="text", dtype=self.DataType.VARCHAR, max_length=65535),
            self.FieldSchema(name="embedding", dtype=self.DataType.FLOAT_VECTOR, dim=dim)
        ]
        schema = self.CollectionSchema(fields, "RAG知识库集合")
        collection = self.Collection(name=collection_name, schema=schema)
        index_params = {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 1024}}
        collection.create_index(field_name="embedding", index_params=index_params)
        return collection

    def insert(self, collection_name: str, texts: List[str], vectors: List[List[float]]):
        collection = self.create_or_get_collection(collection_name)
        collection.insert([texts, vectors])
        collection.flush()

    def search(self, collection_name: str, query_vector: List[List[float]], top_k: int = 10) -> List[str]:
        if not self.utility.has_collection(collection_name):
            return ["错误：知识库集合不存在。"]
        collection = self.Collection(collection_name)
        collection.load()
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = collection.search(data=query_vector, anns_field="embedding", param=search_params, limit=top_k, output_fields=["text"])
        return [hit.entity.get('text') for hit in results[0]] if results else []

RELEVANCE_ASSESSMENT_TEMPLATE = """你是一个文档相关性评估员。请判断下面提供的【文档片段】是否能帮助回答【用户问题】。请只回答“是”或“否”。【用户问题】\n{question}\n\n【文档片段】\n---\n{document}\n---\n\n【该文档是否相关？】"""
QUERY_OPTIMIZATION_TEMPLATE = """你是一个搜索引擎优化专家。当前的用户问题在知识库中没有检索到相关的结果。请你换一个角度，使用不同的关键词或表达方式，重新生成一个与原问题意图相同，但可能更容易在数据库中匹配到内容的新问题。请只提供优化后的新问题，不要添加任何解释。【原始问题】\n{question}\n\n【优化后的新问题】"""
FINAL_ANSWER_TEMPLATE = """你是一个专业、严谨的问答助手。请根据下面提供的【可参考的上下文】来回答用户的【问题】。你的回答必须遵循以下规则：1. 完全基于提供的上下文进行回答，禁止使用任何外部知识或进行猜测。2. 在回答中，你必须明确引用信息来源。引用格式为：[来源: 文件名 (块号: X)]。3. 如果上下文内容足以回答问题，请清晰、准确地组织答案。4. 如果上下文内容不相关或不足以回答问题，请明确指出：“根据您提供的文档，我无法找到关于这个问题的确切信息。”5. 回答时请保持客观、专业的口吻，并且总是使用中文。【可参考的上下文】\n---\n{context}\n---\n\n【问题】\n{question}\n\n【你的回答】"""
HYDE_PROMPT_TEMPLATE = """你是一个善于回答问题的助手。请根据用户的【问题】，生成一个详细、完整、看起来非常专业的回答。重要提示：这个回答是用于后续检索的，所以它不需要保证事实的绝对正确性，但必须与问题高度相关，并且在格式和措辞上像一篇真实的文档片段。【问题】\n{question}\n\n【请生成一个假想的、用于检索的答案】"""

def fetch_internet_search_results(query: str, num_results: int = 5) -> List[Dict]:
    logging.info(f"🌐 正在执行联网搜索: '{query}'")
    search_results = []
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query=query, region='wt-wt', safesearch='off', timelimit='y', max_results=num_results))
            urls = [r['href'] for r in results]
    except Exception as e:
        logging.error(f"联网搜索失败: {e}")
        return []
    def scrape_url(url: str):
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            response = requests.get(url, timeout=10, headers=headers)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                text = re.sub(r'\s+', ' ', soup.get_text()).strip()
                if text:
                    return {'name': f"Web: {url}", 'content': text}
        except Exception as e:
            logging.warning(f"爬取URL失败: {url}, 原因: {e}")
        return None
    for url in urls:
        scraped_data = scrape_url(url)
        if scraped_data and scraped_data['content']:
            search_results.append(scraped_data)
            logging.info(f"✅ 成功爬取: {url}")
    logging.info(f"🌐 联网搜索完成，获得 {len(search_results)} 个有效网页内容。")
    return search_results

milvus_client = MilvusClient(host=MILVUS_HOST, port=MILVUS_PORT)

def execute_rag_pipeline_ray(files_data: List[Dict], query: str, use_hyde: bool) -> Dict:
    logging.info("🚀 ======== 开始执行RAG工作流 (Qwen模型版) ========")
    logging.info("正在本地实例化模型...")
    embedding_model = EmbeddingActor()
    reranker_model = RerankerActor()
    llm_model = LLMActor()
    logging.info("✅ 模型实例化完成。")
    parse_tasks = [parse_and_chunk_document.remote(f['content'], f['name']) for f in files_data]
    
    hypothetical_answer = ""
    if use_hyde:
        hyde_prompt = HYDE_PROMPT_TEMPLATE.format(question=query)
        hypothetical_answer = llm_model.generate(hyde_prompt).strip()
        retrieval_text = hypothetical_answer
    else:
        retrieval_text = query
    
    parsed_results = ray.get(parse_tasks)
    all_chunks_with_source = [chunk for result in parsed_results for chunk in result]
    if not all_chunks_with_source:
        return {"answer": "❌ 未能从任何文件中提取文本块。", "hypothetical_answer": "", "sources": []}
    all_chunk_texts = [chunk['content'] for chunk in all_chunks_with_source]
    
    current_query = query
    
    for attempt in range(MAX_OPTIMIZATION_ATTEMPTS + 1):
        logging.info(f"--- 第 {attempt + 1} 次尝试 ---")
        if attempt > 0:
            retrieval_text = current_query
        
        logging.info(f"当前用于检索的文本: '{retrieval_text[:100]}...'")
        query_vector = embedding_model.embed([retrieval_text])
        
        if attempt == 0:
            chunk_vectors = embedding_model.embed(all_chunk_texts)
            collection_name = f"rag_session_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            milvus_client.insert(collection_name, all_chunk_texts, chunk_vectors)
        
        retrieved_docs = milvus_client.search(collection_name, query_vector)
        # rerank
        if retrieved_docs:
            logging.info(f"初步检索到 {len(retrieved_docs)} 篇文档，正在进行重排序...")
            rerank_pairs = [(current_query, doc) for doc in retrieved_docs]
            rerank_scores = reranker_model.compute_score(rerank_pairs)
            
            reranked_results = sorted(zip(rerank_scores, retrieved_docs), key=lambda x: x[0], reverse=True)
            
            top_k_reranked = 3
            final_docs = [doc for score, doc in reranked_results[:top_k_reranked]]
            logging.info(f"重排序完成，选出 Top-{top_k_reranked} 篇最相关的文档。")

            if final_docs:
                relevant_docs_with_source = [chunk for chunk in all_chunks_with_source if chunk['content'] in final_docs]
                context_parts = [f"[来源: {doc['source']} (块号: {doc['chunk_index']})]\n{doc['content']}" for doc in relevant_docs_with_source]
                context = "\n---\n".join(context_parts)
                
                final_prompt = FINAL_ANSWER_TEMPLATE.format(question=query, context=context)
                final_response = llm_model.generate(final_prompt)
                
                logging.info("🏁 ======== Ray RAG 工作流执行完毕 ========")
                sources_used = sorted(list(set([doc['source'] for doc in relevant_docs_with_source])))
                return {"answer": final_response, "hypothetical_answer": hypothetical_answer, "sources": sources_used}
        
        if attempt < MAX_OPTIMIZATION_ATTEMPTS:
            logging.warning("未找到相关文档，正在尝试优化查询...")
            optimization_prompt = QUERY_OPTIMIZATION_TEMPLATE.format(question=current_query)
            optimized_query = llm_model.generate(optimization_prompt).strip()
            if optimized_query and optimized_query != current_query:
                current_query = optimized_query
            else:
                break

    return {
        "answer": "抱歉，在您提供的文档中，我多次尝试后仍未找到能回答您问题的相关信息。",
        "hypothetical_answer": hypothetical_answer,
        "sources": []
    }

def run_streamlit_app():
    st.set_page_config(page_title="分布式RAG应用 (Ray版)", layout="wide")
    st.title("🚀 分布式RAG应用-Qwen")
    st.markdown("上传文件、输入问题，系统将通过 Ray 分布式后端并行处理数据并生成回答。")
    if "response" not in st.session_state: st.session_state.response = "请在下方提交问题和文件..."
    if "hypothetical_answer" not in st.session_state: st.session_state.hypothetical_answer = ""
    if "sources" not in st.session_state: st.session_state.sources = []
    with st.sidebar:
        st.subheader("⚙️ 高级选项")
        use_hyde = st.toggle("启用HyDE策略", value=True)
    with st.form("rag_form"):
        query = st.text_input("请输入你的问题:")
        uploaded_files = st.file_uploader("上传知识库文件", accept_multiple_files=True)
        col1, col2, _ = st.columns([1, 1, 3])
        with col1: submit_button = st.form_submit_button("仅文件问答")
        with col2: submit_with_internet_button = st.form_submit_button("文件+联网问答")
    if submit_button or submit_with_internet_button:
        if query:
            all_files_data = []
            if uploaded_files:
                all_files_data.extend([{'name': f.name, 'content': f.getvalue()} for f in uploaded_files])
            if submit_with_internet_button:
                with st.spinner("正在进行联网搜索..."):
                    internet_data = fetch_internet_search_results(query)
                    internet_data_bytes = [
                        {'name': item['name'], 'content': item['content'].encode('utf-8')}
                        for item in internet_data
                    ]
                    all_files_data.extend(internet_data)
            if not all_files_data:
                st.error("错误：请至少上传一个文件或使用联网功能。")
            else:
                with st.spinner("系统正在通过 Ray 分布式后端处理中..."):
                    try:
                        result_dict = execute_rag_pipeline_ray(files_data=all_files_data, query=query, use_hyde=use_hyde)
                        st.session_state.response = result_dict.get("answer", "未能获取回答。")
                        st.session_state.hypothetical_answer = result_dict.get("hypothetical_answer", "")
                        st.session_state.sources = result_dict.get("sources", [])
                    except Exception as e:
                        st.error(f"处理过程中发生严重错误: {e}")
                        logging.error(f"Streamlit UI层捕获到异常: {e}", exc_info=True)
        else:
            st.error("错误：请确保您已输入问题。")
    if st.session_state.hypothetical_answer:
        with st.expander("🔍 查看“慢思考”过程 (HyDE生成的假想答案)"):
            st.info(st.session_state.hypothetical_answer)
    st.subheader("模型的回答:")
    st.text_area("response_output", value=st.session_state.response, height=300, disabled=True, label_visibility="collapsed")
    if st.session_state.sources:
        st.subheader("信息来源:")
        for source in st.session_state.sources:
            st.info(f"📄 {source}")

if __name__ == "__main__":
    run_streamlit_app()