from flask import Flask
from mindnlp.transformers import AutoTokenizer, AutoModelForCausalLM
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from app.ppo.touristAttractions import TouristAttractionEnv
from mindspore import context
import mindspore as ms
from mindspore.communication import get_group_size

# 全局变量定义
model = None
tokenizer = None
env = None
model_ppo = None


def create_app():
    app = Flask(__name__)
    app.config['DEBUG'] = False

    # 设置MindSpore
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    context.set_auto_parallel_context(
        parallel_mode=context.ParallelMode.AUTO_PARALLEL,  # 自动并行模式
        gradients_mean=True,
        device_num=2  # 指定使用2张卡
    )
    # 初始化分布式通信（需在多进程/多线程启动时调用）
    # ms.communication.init()  # 初始化HCCL通信

    # 初始化函数 - 只执行一次
    with app.app_context():
        init_extensions(app)

    # 注册蓝图
    from . import routes
    app.register_blueprint(routes.bp)

    return app


def init_extensions(app):
    """初始化扩展，确保只执行一次"""
    global model, tokenizer, env, model_ppo

    # 加载语言模型
    if model is None:
        print("正在加载Qwen2-7B模型...")
        DEFAULT_CKPT_PATH = '/home/ma-user/work/Qwen2-7B-Instruct'
        model = AutoModelForCausalLM.from_pretrained(
            DEFAULT_CKPT_PATH,
            device_map="balanced",
            load_in_8bit=True,  # 8位量化
            low_cpu_mem_usage=True
        ).npu().eval()
        model.set_train(False)
        print("Qwen2-7B模型加载完成！")

    # 加载分词器
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(DEFAULT_CKPT_PATH)

    # 加载强化学习环境
    if env is None:
        from app.ppo.touristAttractions import TouristAttractionEnv
        env = TouristAttractionEnv(max_attractions=25)

    # 加载PPO模型
    if model_ppo is None:
        print("正在加载PPO模型...")
        model_path = "./app/ppo/flexible_tourist_model"
        model_ppo = PPO.load(model_path, env=env)
        print("PPO模型加载完成！")

    # 存储到app配置中
    app.config['MODEL'] = model
    app.config['TOKENIZER'] = tokenizer
    app.config['ENV'] = env
    app.config['MODEL_PPO'] = model_ppo