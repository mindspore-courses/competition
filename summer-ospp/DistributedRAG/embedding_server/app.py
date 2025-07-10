# embedding_server/app.py

import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np

# ------------------- 模型加载 -------------------
# 这是一个重量级操作，我们希望它在服务启动时只执行一次。
# 因此，我们将模型加载放在全局作用域。
try:
    from mindnlp.sentence import SentenceTransformer
    print("正在加载Embedding模型 (BAAI/bge-base-zh-v1.5)...")
    embedding_model = SentenceTransformer('BAAI/bge-base-zh-v1.5')
    print("✅ Embedding模型加载成功!")
    IS_MOCK = False
except ImportError:
    print("❌ 警告：未找到mindnlp库或加载失败，将使用模拟模型。")
    # 创建一个模拟模型，以便在没有完整环境时也能测试API连通性
    class MockEmbeddingModel:
        """
        模拟的Embedding模型，仅用于测试API连通性。
        encode方法返回随机向量，维度与真实模型一致。
        """
        def encode(self, texts, normalize_embeddings=True):
            print("警告: 正在使用模拟Embedding模型！")
            # 保持返回numpy数组的格式，与真实模型一致
            return np.random.rand(len(texts), 768)
    embedding_model = MockEmbeddingModel()
    IS_MOCK = True
except Exception as e:
    # 捕获除ImportError外的其他异常，防止服务启动失败
    print(f"❌ 加载Embedding模型时发生未知错误: {e}")
    embedding_model = None
    IS_MOCK = True

# ------------------- API 定义 -------------------

# 初始化FastAPI应用
app = FastAPI(
    title="Embedding Service",
    description="一个专门用于文本向量化的微服务，基于MindNLP。",
    version="1.0.0"
)

# 定义API请求体的数据模型
class EmbeddingRequest(BaseModel):
    texts: List[str]

# 定义API响应体的数据模型
class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]

@app.post("/embed", response_model=EmbeddingResponse, summary="文本向量化")
async def get_embeddings(request: EmbeddingRequest):
    """
    接收一个包含多个文本字符串的列表，返回它们对应的向量列表。
    """
    if not embedding_model:
        raise HTTPException(status_code=500, detail="Embedding模型未能成功加载，服务不可用。")
    
    try:
        print(f"收到 {len(request.texts)} 条文本的向量化请求...")
        # 调用模型的encode方法进行批量向量化
        vectors = embedding_model.encode(request.texts, normalize_embeddings=True)
        
        # --- 逻辑修正 ---
        # 无论真实模型还是模拟模型，现在都返回numpy数组，可以直接转换
        vectors_list = vectors.tolist()
        
        print("✅ 向量化完成。")
        return EmbeddingResponse(embeddings=vectors_list)
    except Exception as e:
        logging.error(f"向量化过程中发生错误: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"向量化失败: {str(e)}")

@app.get("/health", summary="健康检查")
async def health_check():
    """
    一个简单的健康检查端点，用于确认服务是否正在运行。
    """
    return {"status": "ok" if embedding_model else "error", "model_loaded": not IS_MOCK}
