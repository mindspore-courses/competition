from flask import Flask
import torch
import torch_npu
from transformers import AutoModelForCausalLM, AutoTokenizer
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from app.ppo.touristAttractions import TouristAttractionEnv

model = None

def create_app():
    app = Flask(__name__)
    app.config['DEBUG'] = True
    # 设置设备和其他配置
    global torch_device
    torch_device = "npu:1"
    torch.npu.set_device(torch.device(torch_device))
    torch.npu.set_compile_mode(jit_compile=False)
    option = {}
    option["NPU_FUZZY_COMPILE_BLACKLIST"] = "Tril"
    torch.npu.set_option(option)

    # 加载模型和分词器
    DEFAULT_CKPT_PATH = './app/models/Qwen2-7B-Instruct'
    global model
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(
            DEFAULT_CKPT_PATH,
            torch_dtype=torch.float16,
            device_map=torch_device
        ).npu().eval()
        # 将模型封装为 DataParallel
        if torch.npu.device_count() > 1:
            model = torch.nn.DataParallel(model)
    # 将模型注册到应用上下文
    @app.before_request
    def before_first_request():
        """确保模型在第一个请求前已加载"""
        app.config['MODEL'] = model
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_CKPT_PATH)
    # 加载强化学习模型
    # 初始化环境
    global env
    env = TouristAttractionEnv(max_attractions=25)
    # 模型文件路径
    model_path = "./app/ppo/flexible_tourist_model"  # 修改为你的模型路径
    # 加载模型
    print("正在加载模型...")
    global model_ppo
    model_ppo = PPO.load(model_path, env=env)
    print("模型加载成功！")
    from . import routes
    app.register_blueprint(routes.bp)

    return app