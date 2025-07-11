# llm_server/app.py

import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict

# ------------------- 模型加载 -------------------
try:
    # 导入必要的库
    import mindspore
    from mindnlp.transformers import AutoTokenizer, AutoModelForCausalLM
    LLM_MODEL_PATH = 'openbmb/MiniCPM-2B-dpo-bf16'
    # LLM_MODEL_PATH = 'openbmb/MiniCPM-S-1B-sft'
    # LLM_MODEL_PATH = 'openbmb/MiniCPM-2B-dpo-int4'
    
    print(f"正在加载LLM模型及分词器 ({LLM_MODEL_PATH})...")
    print("这可能需要几分钟时间，并消耗大量内存和网络带宽。请耐心等待。")

    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_PATH, mirror="huggingface")
    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_PATH, ms_dtype=mindspore.float16, mirror="huggingface")
    
    print("✅ LLM模型加载成功!")

except ImportError:
    print("❌ 警告：未找到mindspore或mindnlp库，将使用模拟模型进行API连通性测试。")
    # 创建一个模拟模型，用于在没有完整MindSpore环境时也能测试API连通性
    class MockLLM:
        """
        模拟的大语言模型，仅用于测试API连通性。
        chat方法返回固定格式的模拟回复。
        """
        def chat(self, tokenizer, prompt, history, max_length):
            print("警告: 正在使用模拟LLM模型！")
            return f"模拟回答: 这是对 '{prompt[:50]}...' 的回复。", []
    model = MockLLM()
    tokenizer = None # 模拟模式下分词器为空
except Exception as e:
    # 捕获除ImportError外的其他异常，防止服务启动失败
    print(f"❌ 加载LLM模型时发生严重错误: {e}")
    model = None
    tokenizer = None

# ------------------- API 定义 -------------------

# 初始化FastAPI应用，并添加文档信息
app = FastAPI(
    title="LLM Inference Service",
    description="一个专门用于大语言模型推理的微服务，基于MindNLP和MindSpore。",
    version="1.0.0"
)

# 定义API请求体的数据模型，使用Pydantic进行数据校验
class GenerationRequest(BaseModel):
    prompt: str
    history: List[Dict] = []
    max_length: int = 1024

# 定义API响应体的数据模型
class GenerationResponse(BaseModel):
    response: str

@app.post("/generate", response_model=GenerationResponse, summary="生成回答")
async def generate_response(request: GenerationRequest):
    """
    接收一个包含prompt的请求，调用加载好的LLM生成回答。
    这是本服务的核心功能。
    """
    # 检查模型是否已成功加载
    if not model:
        raise HTTPException(status_code=503, detail="LLM模型未能成功加载，服务当前不可用。")

    try:
        print(f"收到生成请求，Prompt长度: {len(request.prompt)}")
        response_text, _ = model.chat(
            tokenizer,
            request.prompt,
            history=request.history,
            max_length=request.max_length,
        )
        
        print("✅ 回答生成完成。")
        # 将生成的文本封装在响应模型中返回
        return GenerationResponse(response=response_text)
    except Exception as e:
        # 记录详细的错误日志，方便排查问题
        logging.error(f"LLM推理过程中发生错误: {e}", exc_info=True)
        # 向客户端返回一个通用的服务器错误信息
        raise HTTPException(status_code=500, detail=f"推理失败: {str(e)}")

@app.get("/health", summary="健康检查")
async def health_check():
    """
    一个简单的健康检查端点，用于外部系统（如Docker Compose的healthcheck）
    监控本服务是否正在正常运行。
    """
    if model:
        return {"status": "ok", "message": "LLM服务正常运行，模型已加载。"}
    else:
        return {"status": "error", "message": "LLM服务异常，模型未能加载。"}

