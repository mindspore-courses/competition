import ray
import re
import PyPDF2
import markdown
from bs4 import BeautifulSoup
import tiktoken
from typing import List

from io import BytesIO
import logging

# from rapidocr_onnxruntime import RapidOCR
# from PIL import Image
# ==============================================================================
# 1. 定义 Embedding Actor
# ==============================================================================
@ray.remote(num_cpus=4)  
class EmbeddingActor:
    """
    一个专用于文本向量化的 Ray Actor。
    它在自己的进程中加载并持有 BAAI/bge-base-zh-v1.5 模型。
    """
    def __init__(self):

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
# ==============================================================================
@ray.remote(num_cpus=6)  
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
            self.model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_PATH, ms_dtype=mindspore.float16, mirror="huggingface")
            
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
# ==============================================================================
@ray.remote
def parse_and_chunk_document(file_content: bytes, file_name: str) -> List[str]:
    """
    一个 Ray Task，用于解析单个文件内容并将其分块。
    现在已集成OCR功能，可以处理图片。
    """
    print(f"⚙️ Ray Task: 正在解析文件 '{file_name}'...")
    from rapidocr_onnxruntime import RapidOCR
    from PIL import Image
    import numpy as np
    # [新增] 初始化OCR引擎。这个对象可以被复用。
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

    # [新增] 处理图片的核心函数
    def read_image(stream) -> str:
        # 使用Pillow打开图像数据流
        img = Image.open(stream)
        # 将Pillow图像对象转为numpy数组以供OCR引擎使用
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

    # [修改] 在文件类型判断中加入对常见图片格式的支持
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

    # 使用 tiktoken进行分块 
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    chunks = []
    max_token_len = 600
    cover_content = 150
    i = 0
    while i < len(tokens):
        end = min(i + max_token_len, len(tokens))
        chunk_text = enc.decode(tokens[i:end])
        chunks.append(chunk_text)
        i += (max_token_len - cover_content)
        
    print(f"✅ Ray Task: 文件 '{file_name}' 解析并分块为 {len(chunks)} 块。")
    return chunks  