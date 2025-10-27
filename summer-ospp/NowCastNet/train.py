import os
import datetime
import argparse
import random
from pathlib import Path

import numpy as np
import mindspore as ms
from mindspore import context, nn, set_seed
import mindspore.train as train

from nowcasting.models.model_factory import Model
from nowcasting.models.nowcastnet import Net
from nowcasting.data_provider.datasets_factory import RadarData, NowcastDataset
from nowcasting.loss import GenerateLoss, DiscriminatorLoss, EvolutionLoss
from nowcasting.layers.generation.discriminators import TemporalDiscriminator
from nowcasting.layers.generation.generative_network import GenerationNet
from nowcasting.layers.evolution.evolution_network import EvolutionNet
from nowcasting.trainer import EvolutionTrainer, GenerationTrainer

np.random.seed(0)
set_seed(0)
random.seed(0)


class Config:
    """配置类，用于存储训练参数"""
    def __init__(self, **kwargs):
        # 数据参数
        self.dataset_name = kwargs.get('dataset_name', 'radar')
        self.dataset_path = kwargs.get('dataset_path', 'dataset')
        self.input_length = kwargs.get('input_length', 9)
        self.total_length = kwargs.get('total_length', 29)
        self.img_height = kwargs.get('img_height', 512)
        self.img_width = kwargs.get('img_width', 512)
        self.img_ch = kwargs.get('img_ch', 2)
        self.batch_size = kwargs.get('batch_size', 1)
        self.num_workers = kwargs.get('num_workers', 1)
        self.data_frequency = kwargs.get('data_frequency', 10)
        
        # 模型参数
        self.model_name = kwargs.get('model_name', 'NowcastNet')
        self.ngf = kwargs.get('ngf', 32)
        self.noise_scale = kwargs.get('noise_scale', 32)
        
        # 训练参数
        self.epochs = kwargs.get('epochs', 200)
        self.g_lr = kwargs.get('g_lr', 1.5e-5)
        self.d_lr = kwargs.get('d_lr', 6e-5)
        self.evo_lr = kwargs.get('evo_lr', 5e-4)
        self.beta1 = kwargs.get('beta1', 0.5)
        self.beta2 = kwargs.get('beta2', 0.999)
        self.weight_decay = kwargs.get('weight_decay', 0.1)
        self.motion_lambda = kwargs.get('motion_lambda', 1e-2)
        
        # 其他参数
        self.device_target = kwargs.get('device_target', 'Ascend')
        self.device_id = kwargs.get('device_id', 0)
        self.distribute = kwargs.get('distribute', False)
        self.save_checkpoint_epochs = kwargs.get('save_checkpoint_epochs', 10)
        self.eval_interval = kwargs.get('eval_interval', 10)
        self.ckpt_save_dir = kwargs.get('ckpt_save_dir', 'checkpoints')
        self.log_dir = kwargs.get('log_dir', 'logs')
        
        # 计算派生参数
        self.evo_ic = self.total_length - self.input_length
        self.gen_oc = self.total_length - self.input_length
        self.ic_feature = self.ngf * 10
        self.pool_ensemble_num = kwargs.get('pool_ensemble_num', 4)
        
        # 创建保存目录
        Path(self.ckpt_save_dir).mkdir(exist_ok=True)
        Path(self.log_dir).mkdir(exist_ok=True)


def get_args():
    """获取用户指定的参数"""
    parser = argparse.ArgumentParser(description='NowcastNet Training Script')
    
    # 基本参数
    parser.add_argument('--device_target', '-d', type=str, default="Ascend",
                       help='设备类型')
    parser.add_argument("--mode", type=str, default="PYNATIVE", 
                       choices=["GRAPH", "PYNATIVE"],
                       help="Context mode, support 'GRAPH', 'PYNATIVE'")
    parser.add_argument('--device_id', type=int, default=0,
                       help='设备ID')
    parser.add_argument('--run_mode', type=str, choices=["train", "test", "evolution", "generation"], 
                       default='train',
                       help='运行模式: train(完整训练), evolution(演化模块), generation(生成模块), test(测试)')
    
    # 数据参数
    parser.add_argument('--dataset_name', type=str, default='radar',
                       help='数据集名称')
    parser.add_argument('--dataset_path', type=str, default='dataset',
                       help='数据集路径')
    parser.add_argument('--input_length', type=int, default=9,
                       help='输入序列长度')
    parser.add_argument('--total_length', type=int, default=29,
                       help='总序列长度')
    parser.add_argument('--img_height', type=int, default=512,
                       help='图像高度')
    parser.add_argument('--img_width', type=int, default=512,
                       help='图像宽度')
    parser.add_argument('--img_ch', type=int, default=2,
                       help='图像通道数')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='批次大小')
    parser.add_argument('--num_workers', type=int, default=1,
                       help='数据加载工作进程数')
    parser.add_argument('--data_frequency', type=int, default=10,
                       help='数据频率')
    
    # 模型参数
    parser.add_argument('--model_name', type=str, default='NowcastNet',
                       help='模型名称')
    parser.add_argument('--ngf', type=int, default=32,
                       help='生成器特征数')
    parser.add_argument('--noise_scale', type=int, default=32,
                       help='噪声缩放因子')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=1,
                       help='训练轮数')
    parser.add_argument('--g_lr', type=float, default=1.5e-5,
                       help='生成器学习率')
    parser.add_argument('--d_lr', type=float, default=6e-5,
                       help='判别器学习率')
    parser.add_argument('--evo_lr', type=float, default=5e-4,
                       help='演化网络学习率')
    parser.add_argument('--beta1', type=float, default=0.5,
                       help='Adam优化器beta1参数')
    parser.add_argument('--beta2', type=float, default=0.999,
                       help='Adam优化器beta2参数')
    parser.add_argument('--weight_decay', type=float, default=0.1,
                       help='权重衰减')
    parser.add_argument('--motion_lambda', type=float, default=1e-2,
                       help='运动正则化权重')
    
    # 其他参数
    parser.add_argument('--distribute', action='store_true',
                       help='是否使用分布式训练')
    parser.add_argument('--save_checkpoint_epochs', type=int, default=10,
                       help='保存检查点的间隔轮数')
    parser.add_argument('--eval_interval', type=int, default=10,
                       help='验证间隔轮数')
    parser.add_argument('--ckpt_save_dir', type=str, default='checkpoints',
                       help='检查点保存目录')
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='日志保存目录')
    parser.add_argument('--pool_ensemble_num', type=int, default=4,
                       help='集成池数量')
    
    return parser.parse_args()


def init_generation_models(config):
    generator = GenerationNet(config)
    
    discriminator = TemporalDiscriminator(
        in_channels=config.total_length, 
        hidden1=64, 
        hidden2=84, 
        hidden3=40
    )
    
    return generator, discriminator


def init_evolution_model(config):
    model = EvolutionNet(
        t_in=config.input_length,
        t_out=config.evo_ic,
        h_size=config.img_height,
        w_size=config.img_width,
        in_channels=32,
        bilinear=True
    )
    return model


def train_evolution(config):
    print("开始训练演化模块...")
    
    model = init_evolution_model(config)
    
    loss_config = {
        'data': {
            't_in': config.input_length,
            't_out': config.evo_ic,
            'h_size': config.img_height,
            'w_size': config.img_width,
        },
        'optimizer-evo': {
            'motion_lambda': config.motion_lambda
        }
    }
    loss_fn = EvolutionLoss(model, loss_config)
    
    loss_scale = train.loss_scale_manager.FixedLossScaleManager(loss_scale=2048)
    
    trainer = EvolutionTrainer(config, model, loss_fn, loss_scale)
    trainer.train()
    
    print("演化模块训练完成!")


def train_generation(config):
    print("开始训练生成模块...")
    generator, discriminator = init_generation_models(config)
    
    g_loss_fn = GenerateLoss(generator, discriminator)
    d_loss_fn = DiscriminatorLoss(generator, discriminator)
    
    loss_scale = nn.DynamicLossScaleUpdateCell(loss_scale_value=2**12, scale_factor=2, scale_window=1000)
    
    trainer = GenerationTrainer(config, generator, discriminator, g_loss_fn, d_loss_fn, loss_scale)
    trainer.train()
    
    print("生成模块训练完成!")

def main():
    args = get_args()
    
    # 直接使用命令行参数创建配置
    config = Config(**vars(args))
    
    # 设置MindSpore上下文
    context.set_context(
        mode=context.GRAPH_MODE if args.mode.upper().startswith("GRAPH") else context.PYNATIVE_MODE,
        device_target=args.device_target,
        device_id=args.device_id
    )
    
    try:
        if args.run_mode == 'evolution':
            train_evolution(config)
        elif args.run_mode == 'generation':
            train_generation(config)
        elif args.run_mode == 'train':
            train_evolution(config)
            train_generation(config)
        else:
            raise ValueError(f"不支持的运行模式: {args.run_mode}")
            
    except Exception as e:
        print(f"训练过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    print(f"结束时间: {datetime.datetime.now()}")
    print("训练脚本执行完成!")
    return 0

if __name__ == '__main__':
    exit_code = main()
    exit(exit_code)