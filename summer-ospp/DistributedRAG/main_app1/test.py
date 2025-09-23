import mindspore
from mindnlp.transformers import AutoTokenizer, AutoModelForCausalLM
LLM_MODEL_PATH = 'openbmb/MiniCPM-2B-dpo-bf16'
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_PATH, mirror="modelscope")
model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_PATH, ms_dtype=mindspore.float32, mirror="modelscope")
