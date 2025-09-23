# 基于mindnlp ray分布式推理
import os
import time
import logging
import re
from datetime import datetime
from typing import List, Dict

import streamlit as st
import ray

from ddgs import DDGS
from bs4 import BeautifulSoup
import requests

from ray_tasks import EmbeddingActor, LLMActor, parse_and_chunk_document

# RAY,MinIO settings
RAY_ADDRESS = os.getenv("RAY_ADDRESS", "ray://127.0.0.1:10001")
MINIO_HOST = os.getenv("MINIO_HOST", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MILVUS_HOST = os.getenv("MILVUS_HOST", "standalone")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MINIO_BUCKET_NAME = "rag-documents"
MAX_OPTIMIZATION_ATTEMPTS = 2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# 1.ray init
try:
    if not ray.is_initialized():
        logging.info(f"正在连接到 Ray 集群: {RAY_ADDRESS}")
        ray.init(address=RAY_ADDRESS, ignore_reinit_error=True)
        logging.info("✅ Ray 连接成功!")
except Exception as e:
    logging.error(f"❌ 无法连接到 Ray 集群: {e}")
    st.error(f"严重错误：无法连接到 Ray 计算集群，请检查 Ray Head 服务是否正常运行。错误详情: {e}")
    st.stop()



# 2. MilvusClient
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

    def create_or_get_collection(self, collection_name: str, dim: int = 768) -> 'Collection':
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

    def search(self, collection_name: str, query_vector: List[List[float]], top_k: int = 3) -> List[str]:
        if not self.utility.has_collection(collection_name):
            return ["错误：知识库集合不存在。"]
        collection = self.Collection(collection_name)
        collection.load()
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = collection.search(data=query_vector, anns_field="embedding", param=search_params, limit=top_k, output_fields=["text"])
        return [hit.entity.get('text') for hit in results[0]] if results else []


# 3. RAG Prompt templates

RELEVANCE_ASSESSMENT_TEMPLATE = """你是一个文档相关性评估员。请判断下面提供的【文档片段】是否能帮助回答【用户问题】。
请只回答“是”或“否”。

【用户问题】
{question}

【文档片段】
---
{document}
---

【该文档是否相关？】
"""

QUERY_OPTIMIZATION_TEMPLATE = """你是一个搜索引擎优化专家。当前的用户问题在知识库中没有检索到相关的结果。
请你换一个角度，使用不同的关键词或表达方式，重新生成一个与原问题意图相同，但可能更容易在数据库中匹配到内容的新问题。
请只提供优化后的新问题，不要添加任何解释。

【原始问题】
{question}

【优化后的新问题】
"""

FINAL_ANSWER_TEMPLATE = """你是一个专业、严谨的问答助手。请根据下面提供的【可参考的上下文】来回答用户的【问题】。
你的回答必须遵循以下规则：
1.  完全基于提供的上下文进行回答，禁止使用任何外部知识或进行猜测。
2.  在回答中，你必须明确引用信息来源。引用格式为：[来源: 文件名 (块号: X)]。
3.  如果上下文内容足以回答问题，请清晰、准确地组织答案。
4.  如果上下文内容不相关或不足以回答问题，请明确指出：“根据您提供的文档，我无法找到关于这个问题的确切信息。”
5.  回答时请保持客观、专业的口吻，并且总是使用中文。

【可参考的上下文】
---
{context}
---

【问题】
{question}

【你的回答】
"""

HYDE_PROMPT_TEMPLATE = """你是一个善于回答问题的助手。请根据用户的【问题】，生成一个详细、完整、看起来非常专业的回答。
重要提示：这个回答是用于后续检索的，所以它不需要保证事实的绝对正确性，但必须与问题高度相关，并且在格式和措辞上像一篇真实的文档片段。

【问题】
{question}

【请生成一个假想的、用于检索的答案】
"""

def fetch_internet_search_results(query: str, num_results: int = 5) -> List[Dict]:
    """使用DuckDuckGo的API进行搜索，并爬取前N个结果的文本内容。"""
    logging.info(f"🌐 正在执行联网搜索: '{query}'")
    search_results = []
    
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(
                query=query,
                region='wt-wt', 
                safesearch='off', 
                timelimit='y', 
                max_results=num_results
            ))
            urls = [r['href'] for r in results]
    except Exception as e:
        logging.error(f"联网搜索失败: {e}")
        return []

    def scrape_url(url: str):
        """爬取单个URL的文本内容。"""
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            response = requests.get(url, timeout=10, headers=headers)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                text = re.sub(r'\s+', ' ', soup.get_text()).strip()
                if text:
                    return {'name': f"Web: {url}", 'content': text}
        except requests.RequestException as e:
            logging.warning(f"爬取URL失败: {url}, 原因: {e}")
        except Exception as e:
            logging.warning(f"处理URL时发生未知错误: {url}, 原因: {e}")
        return None
    for url in urls:
        scraped_data = scrape_url(url)
        if scraped_data and scraped_data['content']:
            search_results.append(scraped_data)
            logging.info(f"✅ 成功爬取: {url}")

    logging.info(f"🌐 联网搜索完成，获得 {len(search_results)} 个有效网页内容。")
    logging.info(search_results[:2])
    return search_results


# 5. rag pipeline

def execute_rag_pipeline_ray(files_data: List[Dict], query: str, use_hyde: bool) -> Dict:
    logging.info("🚀 ======== 开始执行RAG工作流 ========")
    
    try:
        embedding_actor = ray.get_actor("EmbeddingActor")
        llm_actor = ray.get_actor("LLMActor")
        logging.info("✅ Actor 句柄获取成功。")
    except ValueError:
        logging.warning("Actor 未找到，正在创建新的 Actor 实例...")
        embedding_actor = EmbeddingActor.options(name="EmbeddingActor", get_if_exists=True).remote()
        llm_actor = LLMActor.options(name="LLMActor", get_if_exists=True).remote()
        logging.info("✅ 新的 Actor 实例已创建。")

    parse_tasks = [parse_and_chunk_document.remote(f['content'], f['name']) for f in files_data]
    logging.info(f"提交了 {len(parse_tasks)} 个文件解析任务到 Ray。")
    
    hypothetical_answer_ref = None
    if use_hyde:
        logging.info("💡 HyDE策略已启用，正在生成假想答案...")
        hyde_prompt = HYDE_PROMPT_TEMPLATE.format(question=query)
        hypothetical_answer_ref = llm_actor.generate.remote(hyde_prompt)

    parsed_results = ray.get(parse_tasks)
    all_chunks_with_source = [chunk for result in parsed_results for chunk in result]
    
    if not all_chunks_with_source:
        return {"answer": "❌ 未能从任何文件中提取文本块。", "hypothetical_answer": "", "sources": []}

    all_chunk_texts = [chunk['content'] for chunk in all_chunks_with_source]
    logging.info(f"所有文件解析完成，共得到 {len(all_chunk_texts)} 个文本块。:{all_chunk_texts[:2]}...")
    
    hypothetical_answer = ""
    if use_hyde and hypothetical_answer_ref:
        hypothetical_answer = ray.get(hypothetical_answer_ref).strip()
        logging.info(f"📝 生成的假想答案: '{hypothetical_answer[:100]}...'")
        retrieval_text = hypothetical_answer
    else:
        retrieval_text = query

    current_query = query
    
    for attempt in range(MAX_OPTIMIZATION_ATTEMPTS + 1):
        logging.info(f"--- 第 {attempt + 1} 次尝试 ---")
        
        if attempt > 0:
            retrieval_text = current_query

        logging.info(f"当前用于检索的文本: '{retrieval_text[:100]}...'")
        
        query_vector_ref = embedding_actor.embed.remote([retrieval_text])
        chunk_vectors_ref = embedding_actor.embed.remote(all_chunk_texts)
        query_vector, chunk_vectors = ray.get([query_vector_ref, chunk_vectors_ref])
        
        if not chunk_vectors or not query_vector:
            return {"answer": "❌ 向量化失败。", "hypothetical_answer": hypothetical_answer, "sources": []}
        
        if attempt == 0:
            collection_name = f"rag_session_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            milvus_client.insert(collection_name, all_chunk_texts, chunk_vectors)
            logging.info(f"已将 {len(all_chunk_texts)} 个向量存入 Milvus 集合 '{collection_name}'。")

        retrieved_docs = milvus_client.search(collection_name, query_vector)
        
        if retrieved_docs:
            assessment_tasks = [
                llm_actor.generate.remote(
                    RELEVANCE_ASSESSMENT_TEMPLATE.format(question=current_query, document=doc)
                ) for doc in retrieved_docs
            ]
            assessment_results = ray.get(assessment_tasks)
            
            relevant_docs_texts = [doc for doc, assessment in zip(retrieved_docs, assessment_results) if "是" in assessment.strip()]
            logging.info(f"检索到 {len(retrieved_docs)} 篇文档，其中 {len(relevant_docs_texts)} 篇通过相关性评估。")

            if relevant_docs_texts:
                relevant_docs_with_source = [
                    chunk for chunk in all_chunks_with_source if chunk['content'] in relevant_docs_texts
                ]
                
                context_parts = []
                for doc in relevant_docs_with_source:
                    source_tag = f"[来源: {doc['source']} (块号: {doc['chunk_index']})]"
                    context_parts.append(f"{source_tag}\n{doc['content']}")
                context = "\n---\n".join(context_parts)
                
                final_prompt = FINAL_ANSWER_TEMPLATE.format(question=query, context=context)
                logging.info("提交最终答案生成任务。")
                final_response = ray.get(llm_actor.generate.remote(final_prompt))
                logging.info("🏁 ======== Ray RAG 工作流执行完毕 ========")
                
                sources_used = sorted(list(set([doc['source'] for doc in relevant_docs_with_source])))
                return {"answer": final_response, "hypothetical_answer": hypothetical_answer, "sources": sources_used}
        
        if attempt < MAX_OPTIMIZATION_ATTEMPTS:
            logging.warning("未找到相关文档，正在尝试优化查询...")
            optimization_prompt = QUERY_OPTIMIZATION_TEMPLATE.format(question=current_query)
            optimized_query = ray.get(llm_actor.generate.remote(optimization_prompt)).strip()
            if optimized_query and optimized_query != current_query:
                current_query = optimized_query
            else:
                logging.error("查询优化失败，无法生成新的查询。")
                break
        else:
            logging.warning("已达到最大优化次数。")

    return {
        "answer": "抱歉，在您提供的文档中，我多次尝试后仍未找到能回答您问题的相关信息。",
        "hypothetical_answer": hypothetical_answer,
        "sources": []
    }


# 6. Streamlit

def run_streamlit_app():
    st.set_page_config(page_title="分布式RAG应用 (Ray版)", layout="wide")
    st.title("🚀 分布式RAG应用 (Ray 统一计算后端)")
    st.markdown("上传文件、输入问题，可选联网搜索，系统将通过 Ray 分布式后端并行处理数据并生成回答。")

    if "response" not in st.session_state:
        st.session_state.response = "请在下方提交问题和文件，我会在这里给出回答..."
    if "hypothetical_answer" not in st.session_state:
        st.session_state.hypothetical_answer = ""
    if "sources" not in st.session_state:
        st.session_state.sources = []

    with st.sidebar:
        st.subheader("⚙️ 高级选项")
        use_hyde = st.toggle("启用HyDE策略", value=True, help="通过生成假想答案来优化检索，可能提升相关性但会增加少量延迟。")

    with st.form("rag_form"):
        query = st.text_input("请输入你的问题:", placeholder="例如：这份文档的核心内容是什么？")
        
        uploaded_files = st.file_uploader(
            "上传知识库文件（支持Docx, PPTX, Csv, PDF, MD, Txt, 图片, 语音）",
            accept_multiple_files=True,
            type=['docx', 'pptx', 'csv', 'pdf', 'md', 'txt', 'png', 'jpg', 'jpeg', 'wav', 'mp3', 'm4a']
        )
        
        col1, col2, _ = st.columns([1, 1, 3])
        with col1:
            submit_button = st.form_submit_button("仅文件问答")
        with col2:
            submit_with_internet_button = st.form_submit_button("文件+联网问答")


    if submit_button or submit_with_internet_button:
        if query:
            all_files_data = []
            if uploaded_files:
                all_files_data.extend([{'name': f.name, 'content': f.getvalue()} for f in uploaded_files])
            
            if submit_with_internet_button:
                with st.spinner("正在进行联网搜索并爬取内容..."):
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
                        logging.info(f"Streamlit 接收到查询: '{query}' 和 {len(all_files_data)} 个数据源。")
                        result_dict = execute_rag_pipeline_ray(files_data=all_files_data, query=query, use_hyde=use_hyde)
                        
                        st.session_state.response = result_dict.get("answer", "未能获取回答。")
                        st.session_state.hypothetical_answer = result_dict.get("hypothetical_answer", "")
                        st.session_state.sources = result_dict.get("sources", [])
                    except Exception as e:
                        error_message = f"处理过程中发生严重错误: {e}"
                        st.error(error_message)
                        st.session_state.response = error_message
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