import math
import numpy as np
import os

import mindspore as ms
from mindspore import nn, Model, ops, Tensor
from mindspore.train.callback import LossMonitor, TimeMonitor
from mindspore.train.serialization import save_checkpoint

from nowcasting.data_provider.datasets_factory import RadarData, NowcastDataset
from mindspore.train.callback import Callback
from mindspore.train.summary import SummaryRecord


class NowcastCallBack:
    def __init__(self, config, dataset_size=5000):
        self.config = config
        self.dataset_size = dataset_size
        self.output_path = getattr(config, 'ckpt_save_dir', 'checkpoints')
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        self.ckpt_dir = self.output_path
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
            
        self.run_distribute = getattr(config, 'distribute', False)
        self.rank_id = 0
        if self.run_distribute:
            try:
                import mindspore.communication.management as D
                self.rank_id = D.get_rank()
            except:
                self.rank_id = 0
        
        self.epoch = 0
        self.epoch_start_time = None
        self.step = 0
        self.step_start_time = None
        self.batch_size = getattr(config, 'batch_size', 1)
        self.keep_checkpoint_max = getattr(config, 'keep_checkpoint_max', 5)
        self.ckpt_list = []
        self.epoch_times = []

    def epoch_start(self):
        import time
        self.epoch_start_time = time.time()
        self.epoch += 1

    def step_start(self):
        import time
        self.step_start_time = time.time()
        self.step += 1

    def print_loss(self, res_g, res_d, step=False):
        """print log when step end."""
        import time
        loss_d = float(res_d)
        loss_g = float(res_g)
        print("loss_d", loss_d)
        print("loss_g", loss_g)
        losses = "D_loss: {:.3f}, G_loss:{:.3f}".format(loss_d, loss_g)
        if step:
            step_cost = (time.time() - self.step_start_time) * 1000
            info = "epoch[{}] step {}, cost: {:.2f} ms, {}".format(
                self.epoch, self.step, step_cost, losses)
        else:
            epoch_cost = (time.time() - self.epoch_start_time) * 1000
            info = "epoch[{}] epoch cost: {:.2f} ms, {}".format(
                self.epoch, epoch_cost, losses)
        if self.run_distribute:
            info = "Rank[{}] , {}".format(self.rank_id, info)
        print(f"[INFO] {info}")
        if not step:
            self.epoch_start_time = time.time()

    def epoch_end(self):
        """Evaluate the model at the end of epoch."""
        import time
        epoch_cost = (time.time() - self.epoch_start_time) * 1000
        self.epoch_times.append(epoch_cost)
        self.step = 0

    def save_generation_ckpt(self, g_solver, d_solver=None):
        """save the model at the end of epoch."""
        if self.run_distribute:
            ckpt_name = f"generator-device{self.rank_id}"
        else:
            ckpt_name = "generator"
        g_name = os.path.join(self.ckpt_dir, f"{ckpt_name}_{self.epoch}.ckpt")
        
        if hasattr(g_solver.network, 'generator'):
            save_checkpoint(g_solver.network.generator, g_name)
        else:
            save_checkpoint(g_solver.network, g_name)
            
        self.ckpt_list.append(f"{ckpt_name}_{self.epoch}.ckpt")
        if len(self.ckpt_list) > self.keep_checkpoint_max:
            del_ckpt = self.ckpt_list[0]
            try:
                os.remove(os.path.join(self.ckpt_dir, del_ckpt))
            except FileNotFoundError:
                pass
            self.ckpt_list.remove(del_ckpt)
        
        print(f"生成器检查点已保存: {g_name}")

    def summary(self):
        """train summary at the end of epoch."""
        len_times = len(self.epoch_times)
        sum_times = sum(self.epoch_times)
        epoch_times = sum_times / len_times if len_times > 0 else 0
        info = 'total {} epochs, cost {:.2f} ms, pre epoch cost {:.2f}'.format(len_times, sum_times, epoch_times)
        if self.run_distribute:
            info = "Rank[{}] {}".format(self.rank_id, info)
        print(f"[INFO] {info}")
        print('[INFO] ==========end train ===============')


class GenerationTrainer:
    def __init__(self, config, g_model, d_model, g_loss_fn, d_loss_fn, loss_scale):
        self.config = config
        self.noise_scale = getattr(config, 'noise_scale', 32)
        self.epochs = getattr(config, 'epochs', 200)
        self.save_ckpt_epochs = getattr(config, 'save_checkpoint_epochs', 10)
        self.w_size = getattr(config, 'img_width', 512)
        self.h_size = getattr(config, 'img_height', 512)
        self.ngf = getattr(config, 'ngf', 32)
        self.batch_size = getattr(config, 'batch_size', 1)
        self.pool_ensemble_num = getattr(config, 'pool_ensemble_num', 4)
        self.eval_interval = getattr(config, 'eval_interval', 10)
        
        self.loss_scale = loss_scale
        self.g_model = g_model
        self.d_model = d_model
        self.g_loss_fn = g_loss_fn
        self.d_loss_fn = d_loss_fn
        
        self.train_dataset, self.valid_dataset = self.get_dataset()
        self.dataset_size = self.train_dataset.get_dataset_size()
        
        self.g_optimizer, self.d_optimizer = self.get_optimizer()
        
        self.g_solver, self.d_solver = self.get_solver()
        
        self.callback = NowcastCallBack(config, self.dataset_size)
        
        self.g_grad_fn = ms.value_and_grad(self.g_forward_fn, None, self.g_optimizer.parameters)
        self.d_grad_fn = ms.value_and_grad(self.d_forward_fn, None, self.d_optimizer.parameters)
    
    @staticmethod
    def _get_cosine_annealing_lr(lr_init, steps_per_epoch, epochs, eta_min=1e-6):
        total_steps = epochs * steps_per_epoch
        delta = 0.5 * (lr_init - eta_min)
        lr = []
        try:
            for i in range(total_steps):
                tmp_epoch = min(math.floor(i / steps_per_epoch), epochs)
                lr.append(eta_min + delta * (1 + math.cos(math.pi * tmp_epoch / epochs)))
        except ZeroDivisionError:
            return lr
        return lr
    
    def get_dataset(self):
        train_dataset_generator = RadarData(self.config, module_name='generation')
        valid_dataset_generator = RadarData(self.config, module_name='generation')
        
        train_dataset = NowcastDataset(
            train_dataset_generator, 
            module_name='generation',
            distribute=getattr(self.config, 'distribute', False),
            num_workers=getattr(self.config, 'num_workers', 1)
        )
        valid_dataset = NowcastDataset(
            valid_dataset_generator,
            module_name='generation', 
            distribute=getattr(self.config, 'distribute', False),
            num_workers=getattr(self.config, 'num_workers', 1),
            shuffle=False
        )
        
        train_dataset = train_dataset.create_dataset(self.batch_size)
        valid_dataset = valid_dataset.create_dataset(self.batch_size)
        
        return train_dataset, valid_dataset
    
    def get_optimizer(self):
        g_init_lr = getattr(self.config, 'g_lr', 1.5e-5)
        d_init_lr = getattr(self.config, 'd_lr', 6e-5)
        
        g_lr = self._get_cosine_annealing_lr(g_init_lr, self.dataset_size, self.epochs)
        d_lr = self._get_cosine_annealing_lr(d_init_lr, self.dataset_size, self.epochs)
        
        g_optimizer = nn.Adam(
            self.g_model.trainable_params(),
            learning_rate=g_lr,
            beta1=getattr(self.config, 'beta1', 0.5),
            beta2=getattr(self.config, 'beta2', 0.999)
        )
        d_optimizer = nn.Adam(
            self.d_model.trainable_params(),
            learning_rate=d_lr,
            beta1=getattr(self.config, 'beta1', 0.5),
            beta2=getattr(self.config, 'beta2', 0.999)
        )
        
        return g_optimizer, d_optimizer
    
    def get_solver(self):
        g_solver = nn.TrainOneStepWithLossScaleCell(
            self.g_loss_fn, self.g_optimizer, scale_sense=self.loss_scale
        )
        d_solver = nn.TrainOneStepWithLossScaleCell(
            self.d_loss_fn, self.d_optimizer, scale_sense=self.loss_scale
        )
        return g_solver, d_solver
    
    def g_forward_fn(self, inputs, evo_result, real_image, noise, weights):
        loss = self.g_loss_fn(inputs, evo_result, noise, real_image, weights)
        return loss
    
    def d_forward_fn(self, inputs, evo_result, real_image, noise):
        loss = self.d_loss_fn(inputs, evo_result, noise, real_image)
        return loss
    
    def train_step(self, inputs, evo_result, real_image, weights):
        g_noise = Tensor(
            np.random.randn(
                self.batch_size,
                self.ngf,
                self.h_size // self.noise_scale,
                self.w_size // self.noise_scale,
                self.pool_ensemble_num + 1
            ),
            inputs.dtype
        )
        
        d_noise = Tensor(
            np.random.randn(
                self.batch_size,
                self.ngf,
                self.h_size // self.noise_scale,
                self.w_size // self.noise_scale
            ),
            inputs.dtype
        )
        
        # 训练生成器
        g_loss, g_grads = self.g_grad_fn(inputs, evo_result, real_image, g_noise, weights)
        self.g_optimizer(g_grads)
        
        # 训练判别器
        d_loss, d_grads = self.d_grad_fn(inputs, evo_result, real_image, d_noise)
        self.d_optimizer(d_grads)
        
        return g_loss, d_loss
    
    def train(self):
        for epoch in range(self.epochs):
            self.g_loss_fn.set_train(True)
            self.d_loss_fn.set_train(True)
            epoch_g_loss, epoch_d_loss = 0.0, 0.0
            
            self.callback.epoch_start()
            
            for data in self.train_dataset.create_dict_iterator():
                self.callback.step_start()
                inp, evo_result, labels = data.get("inputs"), data.get("evo"), data.get("labels")
                weights = ops.where(labels > 23., 24., labels + 1)
                g_loss, d_loss = self.train_step(inp, evo_result, labels, weights)
                epoch_g_loss += g_loss.asnumpy()
                epoch_d_loss += d_loss.asnumpy()
            
            epoch_g_loss = epoch_g_loss / self.dataset_size
            epoch_d_loss = epoch_d_loss / self.dataset_size
            self.callback.print_loss(epoch_g_loss, epoch_d_loss)
            
            # 定期保存检查点
            if epoch % self.save_ckpt_epochs == 0 or epoch == self.epochs - 1:
                print(f"saving the model at the end of epoch {epoch}")
                self.callback.save_generation_ckpt(self.g_solver, self.d_solver)
                
            self.callback.epoch_end()
        
        self.callback.summary()


class EvolutionTrainer:
    def __init__(self, config, model, loss_fn, loss_scale):
        self.config = config
        self.model = model
        self.loss_fn = loss_fn
        self.loss_scale = loss_scale
        
        self.train_dataset, self.valid_dataset = self.get_dataset()
        self.dataset_size = self.train_dataset.get_dataset_size()
        
        self.optimizer = self.get_optimizer()
        
        self.solver = self.get_solver()
        
        self.pred_cb = self.get_callback()
        self.ckpt_cb = self.pred_cb.save_evolution_ckpt()
    
    def get_dataset(self):
        train_dataset_generator = RadarData(self.config, module_name='evolution')
        valid_dataset_generator = RadarData(self.config, module_name='evolution')
        
        train_dataset = NowcastDataset(
            train_dataset_generator,
            module_name='evolution',
            distribute=getattr(self.config, 'distribute', False),
            num_workers=getattr(self.config, 'num_workers', 1)
        )
        valid_dataset = NowcastDataset(
            valid_dataset_generator,
            module_name='evolution',
            distribute=getattr(self.config, 'distribute', False),
            num_workers=getattr(self.config, 'num_workers', 1),
            shuffle=False
        )
        
        batch_size = getattr(self.config, 'batch_size', 8)
        train_dataset = train_dataset.create_dataset(batch_size)
        valid_dataset = valid_dataset.create_dataset(batch_size)
        
        return train_dataset, valid_dataset
    
    def get_optimizer(self):
        lr = getattr(self.config, 'evo_lr', 5e-4)
        optimizer = nn.Adam(
            self.model.trainable_params(),
            learning_rate=lr,
            weight_decay=getattr(self.config, 'weight_decay', 0.1)
        )
        return optimizer
    
    def get_solver(self):
        solver = Model(
            network=self.loss_fn,
            optimizer=self.optimizer,
            loss_scale_manager=self.loss_scale
        )
        return solver
    
    def get_callback(self):
        class SimpleLogger:
            def info(self, msg):
                print(f"[INFO] {msg}")
        
        logger = SimpleLogger()
        pred_cb = EvolutionCallBack(self.model, self.valid_dataset, self.config, logger)
        return pred_cb
    
    def train(self):
        callback_lst = [LossMonitor(), TimeMonitor(), self.pred_cb, self.ckpt_cb]
        epochs = getattr(self.config, 'epochs', 200)
        data_sink = getattr(self.config, 'data_sink', True)
        
        self.solver.train(
            epoch=epochs,
            train_dataset=self.train_dataset,
            callbacks=callback_lst,
            dataset_sink_mode=data_sink
        )


class EvolutionCallBack(Callback):
    def __init__(self, model, valid_dataset, config, logger):
        super(EvolutionCallBack, self).__init__()
        self.model = model
        self.valid_dataset = valid_dataset
        self.config = config
        self.logger = logger
        
        self.output_path = getattr(config, 'ckpt_save_dir', 'checkpoints')
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        self.ckpt_dir = self.output_path
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
            
        self.predict_interval = getattr(config, 'eval_interval', 10)
        self.epochs = getattr(config, 'epochs', 200)
        self.eval_time = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc_args):
        pass

    def on_train_epoch_end(self, run_context):
        cb_params = run_context.original_args()
        if cb_params.cur_epoch_num % self.predict_interval == 0 or cb_params.cur_epoch_num == self.epochs:
            self.eval_time += 1
            self.logger.info(f"开始第 {self.eval_time} 次验证，当前epoch: {cb_params.cur_epoch_num}")

    def save_evolution_ckpt(self):
        from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
        import mindspore.communication.management as D
        
        distribute = getattr(self.config, 'distribute', False)
        if distribute:
            try:
                rank_id = D.get_rank()
                ckpt_name = f"evolution-device{rank_id}"
            except:
                ckpt_name = "evolution"
        else:
            ckpt_name = "evolution"
            
        ckpt_config = CheckpointConfig(
            save_checkpoint_epochs=getattr(self.config, 'save_checkpoint_epochs', 10),
            keep_checkpoint_max=getattr(self.config, 'keep_checkpoint_max', 5)
        )
        ckpt_cb = ModelCheckpoint(prefix=ckpt_name, directory=self.ckpt_dir, config=ckpt_config)
        return ckpt_cb


def create_evolution_trainer(config, model, loss_fn, loss_scale):
    """创建演化训练器的辅助函数"""
    return EvolutionTrainer(config, model, loss_fn, loss_scale)


def create_generation_trainer(config, g_model, d_model, g_loss_fn, d_loss_fn, loss_scale):
    """创建生成训练器的辅助函数"""
    return GenerationTrainer(config, g_model, d_model, g_loss_fn, d_loss_fn, loss_scale)
