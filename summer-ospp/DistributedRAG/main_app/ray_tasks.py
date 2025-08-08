import ray
import re
import PyPDF2
import markdown
from bs4 import BeautifulSoup
import tiktoken
from typing import List

from io import BytesIO
import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter

# ==============================================================================
# 1. 定义 Embedding Actor
# (逻辑来自原 embedding_server/app.py)
# ==============================================================================
@ray.remote(num_cpus=4)  # 为每个 Embedding Actor 实例分配4个CPU
class EmbeddingActor:
    """
    一个专用于文本向量化的 Ray Actor。
    它在自己的进程中加载并持有 BAAI/bge-base-zh-v1.5 模型。
    """
    def __init__(self):
        # 在 Actor 初始化时加载模型，模型将常驻于该 Actor 的显存中
        try:
            from mindnlp.sentence import SentenceTransformer
            print("▶️ EmbeddingActor: 正在加载模型 (BAAI/bge-base-zh-v1.5)...")
            self.model = SentenceTransformer('BAAI/bge-base-zh-v1.5')
            print("✅ EmbeddingActor: 模型加载成功!")
        except Exception as e:
            print(f"❌ EmbeddingActor: 加载模型失败: {e}")
            self.model = None

    def embed(self, texts: List[str]) -> List[List[float]]:
        """接收文本列表，返回向量列表。这是该 Actor 的核心推理方法。"""
        if not self.model or not texts:
            return []
        
        print(f"⚙️ EmbeddingActor: 正在为 {len(texts)} 条文本生成向量...")
        try:
            vectors = self.model.encode(texts, normalize_embeddings=True)
            return vectors.tolist()
        except Exception as e:
            print(f"❌ EmbeddingActor: 向量化过程中发生错误: {e}")
            return []

# ==============================================================================
# 2. 定义 LLM Actor
# (逻辑来自原 llm_server/app.py)
# ==============================================================================
@ray.remote(num_cpus=6) # 为每个 LLM Actor 实例分配6个CPU
class LLMActor:
    """
    一个专用于大语言模型推理的 Ray Actor。
    它在自己的进程中加载并持有 MiniCPM 模型。
    """
    def __init__(self):
        # 在 Actor 初始化时加载模型和分词器
        try:
            import mindspore
            from mindnlp.transformers import AutoTokenizer, AutoModelForCausalLM
            
            LLM_MODEL_PATH = 'openbmb/MiniCPM-2B-dpo-bf16'
            logging.info(f"▶️ LLMActor: 正在加载LLM模型及分词器 ({LLM_MODEL_PATH})...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_PATH, mirror="huggingface")
            self.model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_PATH, ms_dtype=mindspore.float32, mirror="huggingface")
            
            logging.info("✅ LLMActor: 模型加载成功!")
        except Exception as e:
            logging.info(f"❌ LLMActor: 加载LLM模型时发生严重错误: {e}")
            self.model = None
            self.tokenizer = None

    def generate(self, prompt: str, max_length: int = 1024) -> str:
        """接收格式化后的 prompt，返回生成的文本。"""
        if not self.model or not self.tokenizer:
            return "LLM Actor 模型未成功加载，无法生成回答。"
            
        print(f"⚙️ LLMActor: 收到生成请求，正在调用 model.chat...")
        try:
            response_text, _ = self.model.chat(
                self.tokenizer,
                prompt,
                history=[],
                max_length=max_length,
            )
            return response_text
        except Exception as e:
            print(f"❌ LLMActor: 推理过程中发生错误: {e}")
            return f"LLM Actor 推理失败: {str(e)}"

# ==============================================================================
# 3. 定义文件处理 Task
# (逻辑来自原 main_app/main.py 中的 FileProcessor 类)
# 这是一个无状态的任务，非常适合用 Ray Task 来并行处理。
# ==============================================================================
@ray.remote
def parse_and_chunk_document(file_content: bytes, file_name: str) -> List[str]:
    """
    一个 Ray Task，用于解析单个文件内容并将其分块。
    此版本采用了更智能的、感知内容结构的分块策略。
    """
    print(f"⚙️ Ray Task: 正在解析文件 '{file_name}'...")
    from rapidocr_onnxruntime import RapidOCR
    from PIL import Image
    import numpy as np
    ocr_engine = RapidOCR()

    def read_pdf(stream) -> str:
        reader = PyPDF2.PdfReader(stream)
        return "".join(page.extract_text() for page in reader.pages if page.extract_text())

    def read_markdown(stream) -> str:
        md_text = stream.read().decode('utf-8')
        html = markdown.markdown(md_text)
        soup = BeautifulSoup(html, 'html.parser')
        return re.sub(r'http\S+', '', soup.get_text())

    def read_text(stream) -> str:
        return stream.read().decode('utf-8')

    def read_image(stream) -> str:
        # 使用Pillow打开图像数据流
        img = Image.open(stream)
        import numpy as np
        img_np = np.array(img)
        
        # 调用OCR引擎进行识别
        result, _ = ocr_engine(img_np)
        
        # 将识别出的文本行拼接成一个完整的字符串
        if result:
            text_list = [line[1] for line in result]
            return "\n".join(text_list)
        return ""

    file_stream = BytesIO(file_content)
    text = ""
    file_suffix = file_name.split('.')[-1].lower()

    if file_suffix == 'pdf':
        text = read_pdf(file_stream)
    elif file_suffix == 'md':
        text = read_markdown(file_stream)
    elif file_suffix == 'txt':
        text = read_text(file_stream)
    elif file_suffix in ['png', 'jpg', 'jpeg', 'bmp', 'tiff']:
        text = read_image(file_stream)
    else:
        print(f"⚠️ Ray Task: 不支持的文件类型 '{file_suffix}'，跳过文件 {file_name}。")
        return []

    if not text:
        print(f"ℹ️ Ray Task: 从文件 '{file_name}' 中未提取到文本。")
        return []

    chunks = []
    if file_suffix == 'md':
        # 对 Markdown 文件使用标题分割器
        print(f"✨ Ray Task: 对 Markdown 文件 '{file_name}' 使用标题分割策略。")
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
        md_header_splits = markdown_splitter.split_text(text)
        
        # 对分割后的每个大块，再进行递归分块，防止有超长章节
        # (这部分和下面的递归分块逻辑是复用的)
        chunk_size = 600
        chunk_overlap = 150
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        for split in md_header_splits:
            chunks.extend(text_splitter.split_text(split.page_content))

    else:
        # 对其他类型文件（PDF, TXT, 图片OCR结果）使用递归字符分割器
        print(f"✨ Ray Task: 对文件 '{file_name}' 使用递归字符分割策略。")
        chunk_size = 600
        chunk_overlap = 150
        # 使用tiktoken来计算块长度
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks = text_splitter.split_text(text)
        
    print(f"✅ Ray Task: 文件 '{file_name}' 解析并分块为 {len(chunks)} 块。")
    return chunks