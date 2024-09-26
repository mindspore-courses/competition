from mindspore import nn
from mindspore import Tensor, ops
import mindspore.numpy as mnp
from mindspore import Parameter
from mindspore.nn.probability.distribution import Normal, Distribution, Bernoulli
from config import *
from distribution import *
from loss import *
class SAMSVAEModel(nn.Cell):
    def __init__(
        self,
        n: int,
        n_latent: int,                            # 隐变量的维度
        n_treatments: int,                        # 扰动（例如药物干预或基因编辑）的数量
        n_phenos: int,                            # 表型数据的维度，例如单细胞RNA测序中的基因数量
        mask_prior_prob: float,                   # 扰动掩码mt的先验概率，mt用于控制稀疏性
        embedding_prior_scale: float,             # 扰动嵌入et的先验尺度
        likelihood_key: LIKELIHOOD_KEY_DTYPE,     # 用于确定似然函数类型的关键字（只能是正态分布、泊松分布、负二项分布）
        decoder_n_layers: int,                    # 解码器的层数
        decoder_n_hidden: int,                    # 解码器每层隐藏单元的数量
    ):
        super(SAMSVAEModel, self).__init__()
        self.n_latent = n_latent
        self.n_treatments = n_treatments
        self.n_phenos = n_phenos
        self.likelihood_key = likelihood_key
        self.decoder_n_layers = decoder_n_layers
        self.decoder_n_hidden = decoder_n_hidden
        self.p_E_loc = Tensor(ops.zeros((n_treatments, n_latent)), mindspore.float32)  # 固定为常量
        self.p_E_scale = Tensor(embedding_prior_scale * ops.ones((n_treatments, n_latent)), mindspore.float32)  # 固定为常量
        self.p_mask_probs = Tensor(mask_prior_prob * ops.ones((n_treatments, n_latent)), mindspore.float32)         
        self.decoder = get_likelihood_mlp(   
            likelihood_key=likelihood_key,
            n_input=n_latent,                      # 输入为隐变量维度
            n_output=n_phenos,                     # 输出为表型数据的维度，例如单细胞RNA测序中的基因数量
            n_layers=decoder_n_layers,             
            n_hidden=decoder_n_hidden,
            use_batch_norm=False,                  # 不使用批次归一化
            activation_fn=nn.LeakyReLU,      # 激活函数为LeakyReLU
        )
        # self.stdnormal= ops.StandardNormal()
        self.means = ops.zeros((n, self.n_latent), mindspore.float32)
        self.stds = ops.ones((n, self.n_latent), mindspore.float32)
        self.generative_dists = {
            "p_z_basal": Normal(self.means, self.stds),
            "p_E": Normal(self.p_E_loc, self.p_E_scale),
            "p_mask": Bernoulli(probs=ops.sigmoid(ops.log(self.p_mask_probs)), dtype=mindspore.float32)
        }
        print(self.generative_dists)
    def get_var_keys(self) -> List[str]:
        return ["z_basal", "E", "mask"]

    def construct(
        self,
        D: mindspore.Tensor,           
        condition_values,
        n_particles,       
    ) -> Tuple[Dict[str, Distribution], Dict[str, mindspore.Tensor]]:
        n = D.shape[0]         # 样本数量
        if condition_values is None:              # 如果没有提供condition_value，则初始化为空字典
            condition_values = dict()
        print("62:", condition_values.keys())
        samples = {}
        for k in self.get_var_keys():
            if condition_values.get(k) is not None:       # 检查是否存在条件值
                value = condition_values[k]
                # expand to align with n_particles（调整样本形状）
                if len(value.shape) == 2:
                    value = ops.expand_dims(value, 0).expand((n_particles, -1, -1))
                    # value = value.unsqueeze(0).expand((n_particles, -1, -1))  
                samples[k] = value
            else:
                samples[k] = self.generative_dists[f"p_{k}"].sample((n_particles,))
        print("74:", condition_values.keys())
        z = samples["z_basal"] + ops.matmul(D, samples["E"] * samples["mask"])
        condition_values["library_size"] = 11111
        if self.likelihood_key != "library_nb":
            self.generative_dists["p_x"] = self.decoder(z)
        else:
            self.generative_dists["p_x"] = self.decoder(z, condition_values["library_size"])
        samples["x"] = self.generative_dists["p_x"].sample()
        return self.generative_dists, samples
    

def get_likelihood_mlp(
    likelihood_key: LIKELIHOOD_KEY_DTYPE,
    n_input: int,
    n_output: int,
    n_layers: int,
    n_hidden: int,
    use_batch_norm: bool,
    activation_fn: nn.Cell = nn.LeakyReLU,
) -> nn.Cell:

    mlp_class: nn.Cell
    if likelihood_key == "normal":
        mlp_class = GaussianLikelihoodResidualMLP
    else:
        mlp_class = LibraryGammaPoissonSharedConcentrationResidualMLP

    mlp = mlp_class(
        n_input=n_input,
        n_output=n_output,
        n_layers=n_layers,
        n_hidden=n_hidden,
        use_batch_norm=use_batch_norm,
        activation_fn=activation_fn,
    )
    return mlp


class BaseGaussianLikelihoodMLP(nn.Cell):
    def __init__(self, mlp, mean_encoder, log_var_encoder, var_eps=1e-6):
        super(BaseGaussianLikelihoodMLP, self).__init__()
        self.mlp = mlp
        self.mean_encoder = mean_encoder
        self.log_var_encoder = log_var_encoder
        self.var_eps = var_eps
        self.stdnormal = ops.StandardNormal()

    def construct(self, x: mindspore.Tensor):
        multiple_particles = len(x.shape) == 3
        if multiple_particles:
            n_particles, n, x_dim = x.shape       #shape = p*N*dim
            x = x.reshape(n_particles * n, x_dim) #shape = (p*N)*dim
        else:
            n_particles = 1
            n = x.shape[0]  # 设置默认的 n 和 x_dim
            x_dim = x.shape[1]
        
        z = self.mlp(x)
        means = self.mean_encoder(z)
       
        log_var = self.log_var_encoder(z)
        vars = ops.exp(log_var) + self.var_eps

        if multiple_particles:
            means = means.reshape(n_particles, n, -1)
            vars = vars.reshape(n_particles, n, -1)
        
        stds = ops.sqrt(vars)
        eps = self.stdnormal(means.shape)
        sample = means + stds * eps
        ret_dict = {
            "sample": sample, # sample tensor
            "means": means,
            "stds": stds,
            "log_var": log_var,
            "vars": vars,
        }
        return ret_dict


class GaussianLikelihoodResidualMLP(BaseGaussianLikelihoodMLP):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_layers: int,
        n_hidden: int,
        use_batch_norm: bool,
        activation_fn: nn.Cell = nn.LeakyReLU,
        use_activation: bool = True,
        var_eps: float = 1e-4,
        
    ):
        super().__init__(mlp=None, mean_encoder=None, log_var_encoder=None)
        self.mlp = ResidualMLP(
            n_input=n_input,
            n_output=n_hidden,
            n_layers=n_layers - 1,
            n_hidden=n_hidden,
            use_batch_norm=use_batch_norm,
            activation_fn=activation_fn,
            use_activation=use_activation,
            last_layer_activation=True,
            last_layer_residual=True,
        )
        self.mean_encoder = nn.Dense(n_hidden, n_output)
        self.log_var_encoder = nn.Dense(n_hidden, n_output)
        self.var_eps = var_eps


class MLP(nn.Cell):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_layers: int,
        n_hidden: int,
        use_batch_norm: bool,
        activation_fn: nn.Cell = nn.LeakyReLU,
        use_activation: bool = True,
        last_layer_activation: bool = True,
    ):
        super().__init__()
        layer_dims = [n_input] + n_layers * [n_hidden] + [n_output]
        layers = []
        for i in range(1, len(layer_dims)):
            skip_activation = (not last_layer_activation) and (i == len(layer_dims) - 1)
            layer_in = layer_dims[i - 1]
            layer_out = layer_dims[i]
            sublayers = [
                nn.Dense(layer_in, layer_out),
                nn.BatchNorm1d(num_features=layer_out)
                if use_batch_norm and not skip_activation
                else None,
                activation_fn() if use_activation and not skip_activation else None,
            ]
            sublayers = [sl for sl in sublayers if sl is not None]
            layer = nn.SequentialCell(*sublayers)
            layers.append(layer)
        self.layers = nn.CellList(layers)

    def construct(self, x):
        x = self.layers[0](x)
        for layer in self.layers[1:]:
            x = layer(x)
        return x


class ResidualMLP(MLP):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_layers: int,
        n_hidden: int,
        use_batch_norm: bool,
        activation_fn: nn.Cell = nn.LeakyReLU,
        use_activation: bool = True,
        last_layer_activation: bool = True,
        last_layer_residual: bool = False,
    ):
        super().__init__(
            n_input,
            n_output,
            n_layers,
            n_hidden,
            use_batch_norm,
            activation_fn,
            use_activation,
            last_layer_activation,
        )
        self.last_layers_residual = last_layer_residual
        assert (not last_layer_residual) or (n_output == n_hidden)

    def construct(self, x):
        for i, layer in enumerate(self.layers):
            if i == 0:
                x = layer(x)
            elif i == len(self.layers) - 1 and not self.last_layers_residual:
                x = layer(x)
            else:
                x = layer(x) + x
        return x

class BaseLibraryGammaPoissonSharedConcentrationLikelihood(nn.Cell): #
    def construct(self, x: mindspore.Tensor, library_size: mindspore.Tensor):
        library_size = mindspore.Tensor(library_size)
        multiple_particles = len(x.shape) == 3
        n_particles, n, x_dim = (0,0,0)
        if multiple_particles:
            n_particles, n, x_dim = x.shape
            x = x.reshape(n_particles * n, x_dim)
            broadcast_shape = (n_particles, library_size.numel())
            library_size = library_size.broadcast_to(broadcast_shape).reshape(-1, 1)
        z = self.mlp(x)
        normalized_mu = self.normalized_mean_decoder(z)
        mu = library_size * normalized_mu

        if multiple_particles:
            mu = mu.reshape(n_particles, n, -1)

        concentration = ops.exp(self.log_concentration)
        mu_eps = 1e-4
        # 阻止泊松分布部分的梯度传播 
        # concentration = ops.stop_gradient(concentration) 
        # mu = ops.stop_gradient(mu)
        dist = GammaPoisson(
            concentration=concentration, rate=concentration / (mu + mu_eps)
        )
        return dist

class LibraryGammaPoissonSharedConcentrationResidualMLP(
    BaseLibraryGammaPoissonSharedConcentrationLikelihood
):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_layers: int,
        n_hidden: int,
        use_batch_norm: bool,
        activation_fn: nn.Cell = nn.LeakyReLU,
        use_activation: bool = True,
    ):
        super().__init__()
        self.mlp = ResidualMLP(
            n_input=n_input,
            n_output=n_hidden,
            n_layers=n_layers - 1,
            n_hidden=n_hidden,
            use_batch_norm=use_batch_norm,
            activation_fn=activation_fn,
            use_activation=use_activation,
            last_layer_activation=True,
            last_layer_residual=True,
        )
        self.normalized_mean_decoder = nn.SequentialCell(
            nn.Dense(n_hidden, n_output),
            nn.Softmax(),
        )
        self.log_concentration = mindspore.Parameter(mindspore.Tensor(mnp.zeros((n_output,)), mindspore.float32), name="log_concentration")


def get_normalization_module(
    key: Literal["standardize", "log_standardize"],
    normalization_stats: ObservationNormalizationStatistics,
):
    module = LogStandardizationModule(normalization_stats)
    return module


class LogStandardizationModule(nn.Cell):
    def __init__(self, normalization_stats: ObservationNormalizationStatistics):
        super().__init__()
        self.log_mean = Parameter(Tensor(normalization_stats.log_x_mean, mindspore.float32), requires_grad=False)
        self.log_scale = Parameter(Tensor(normalization_stats.log_x_std, mindspore.float32), requires_grad=False)

    def construct(self, x):
        logx = ops.log(x + 1)
        return (logx - self.log_mean) / self.log_scale


class SAMSVAEMeanFieldNormalGuide(nn.Cell):
    def __init__(
        self,
        n_latent: int,                      # 隐变量维数
        n_treatments: int,                  # 扰动数量
        n_phenos: int,                      # 表型特征的数量
        basal_encoder_n_layers: int,        # 基态编码器层数
        basal_encoder_n_hidden: int,        # 基态编码器隐藏单元数
        basal_encoder_input_normalization: Optional[        # 基线编码器输入的归一化方式
            Literal["standardize", "log_standardize"]
        ],
        x_normalization_stats: Optional[ObservationNormalizationStatistics],    # 用于归一化的观测统计数据
        embedding_loc_init_scale: float = 0,            # 扰动嵌入的初始位置和尺度
        embedding_scale_init: float = 1,
        mask_init_logits: float = 0,                    # 掩码变量的初始 logits，用于控制扰动的稀疏性
        gs_temperature: float = 1,                      # Gumbel-Softmax 采样的温度参数
        mean_field_encoder: bool = False,               # 是否使用平均场编码器
    ):
        super().__init__()
        self.n_latent = n_latent
        self.n_treatments = n_treatments
        self.n_phenos = n_phenos
        self.basal_encoder_input_normalization = basal_encoder_input_normalization
        self.x_normalization_stats = x_normalization_stats
        self.mean_field_encoder = mean_field_encoder
        q_mask_logits_init = mask_init_logits * mnp.ones((n_treatments, n_latent)) 
        self.q_mask_logits = Parameter(q_mask_logits_init, name="q_mask_logits")
        q_E_loc_init = embedding_loc_init_scale * mnp.randn(n_treatments, n_latent) 
        self.q_E_loc = Parameter(q_E_loc_init, name="q_E_loc") 
        q_E_log_scale_init = mnp.log(float(embedding_scale_init)) * mnp.ones((n_treatments, n_latent)) 
        self.q_E_log_scale = Parameter(q_E_log_scale_init, name="q_E_log_scale")
        self.var_eps = 1e-4
        self.means = self.q_E_loc
        self.scale = ops.exp(self.q_E_log_scale) + self.var_eps
        self.dist = Normal(self.means, self.scale)
        if self.basal_encoder_input_normalization is None:   # 处理输入数据的归一化
            self.normalization_module = None
        else:
            assert x_normalization_stats is not None, "Missing x_normalization_stats"
            self.normalization_module = get_normalization_module(
                key=self.basal_encoder_input_normalization,
                normalization_stats=x_normalization_stats,
            )
        self.z_basal_encoder = get_likelihood_mlp(
            likelihood_key="normal",
            n_input=n_phenos if mean_field_encoder else n_phenos + n_latent,
            n_output=n_latent,
            n_layers=basal_encoder_n_layers,
            n_hidden=basal_encoder_n_hidden,
            use_batch_norm=False,
        )
        self.gs_temperature = gs_temperature
    
    def get_var_keys(self):
        var_keys = ["z_basal", "E", "mask"]
        return var_keys

    def construct(
        self,
        X: Optional[mindspore.Tensor] = None,
        D: Optional[mindspore.Tensor] = None,
        condition_values: Optional[Dict[str, mindspore.Tensor]] = None,
        n_particles: int = 1,
    ) -> Tuple[Dict[str, object], Dict[str, mindspore.Tensor]]:  # object 是 MindSpore 分布类的占位符
        if condition_values is None:
            condition_values = dict()

        guide_distributions: Dict[str, object] = {}
        guide_samples: Dict[str, mindspore.Tensor] = {}
        guide_distributions["q_mask"] = GumbelSoftmaxBernoulliStraightThrough(temperature=self.gs_temperature,logits=self.q_mask_logits)
        guide_distributions["q_E"] = self.dist

        # # pynative error
        # if "mask" not in condition_values:
        #     guide_samples["mask"] = guide_distributions[f"q_mask"].rsample((n_particles,))
        # else:
        #     guide_samples["mask"] = condition_values["mask"]

        # if "E" not in condition_values:
        #     mean = guide_distributions["q_E"].mean()
        #     std = guide_distributions["q_E"].sd()
        #     epsilon = ops.standard_normal((n_particles,) + mean.shape)  # 从标准正态分布采样
        #     guide_samples["E"] = mean + epsilon * std  # 重新参数化采样
        # else:
        #     guide_samples["E"] = condition_values["E"]


        # if X is not None and D is not None:
        #     encoder_input = X
        #     if self.normalization_module is not None:
        #         encoder_input = self.normalization_module(encoder_input)
            
        #     encoder_shape = encoder_input.shape
        #     encoder_input = ops.expand_dims(encoder_input, 0)
        #     encoder_input = ops.broadcast_to(encoder_input, (n_particles, encoder_shape[0], encoder_shape[1]))

        #     if not self.mean_field_encoder:
        #         latent_offset = ops.matmul(D, guide_samples["mask"] * guide_samples["E"])
        #         encoder_input = ops.concat([encoder_input, latent_offset], axis=-1)

        #     dict_q_z_basal = self.z_basal_encoder(encoder_input) # dict
        #     q_z_basal_mean = dict_q_z_basal["means"]
        #     q_z_basal_stds = dict_q_z_basal["stds"]
        #     epsilon = ops.standard_normal(q_z_basal_mean.shape)  # 从标准正态分布采样
        #     guide_samples["z_basal"] = dict_q_z_basal['sample']
        # if "z_basal" in condition_values:
        #     guide_samples["z_basal"] = condition_values["z_basal"]

        return guide_distributions, guide_samples
    
 
class Net(nn.Cell):
    def __init__(self, config, data_module):
        super(Net, self).__init__()
        self.model = SAMSVAEModel(
                n=config["model_kwargs"]["n"],
                n_latent=config["model_kwargs"]["n_latent"],
                n_treatments=config["model_kwargs"]["n_treatments"],
                n_phenos=config["model_kwargs"]["n_phenos"],
                mask_prior_prob=config["model_kwargs"]["mask_prior_prob"],
                embedding_prior_scale=config["model_kwargs"]["embedding_prior_scale"],
                likelihood_key=config["model_kwargs"]["likelihood_key"],
                decoder_n_layers=config["model_kwargs"]["decoder_n_layers"],
                decoder_n_hidden=config["model_kwargs"]["decoder_n_hidden"],
            )

        self.guide = SAMSVAEMeanFieldNormalGuide(
            n_latent=config["guide_kwargs"]["n_latent"],
            n_treatments=config["guide_kwargs"]["n_treatments"],
            n_phenos=config["guide_kwargs"]["n_phenos"],
            basal_encoder_n_layers=config["guide_kwargs"]["basal_encoder_n_layers"],
            basal_encoder_n_hidden=config["guide_kwargs"]["basal_encoder_n_hidden"],
            basal_encoder_input_normalization=config["guide_kwargs"]["basal_encoder_input_normalization"],
            x_normalization_stats=data_module.get_x_train_statistics()
        )
        self.loss_module = SAMSVAE_ELBOLossModule()
        self.loss_fn = self.loss_module.loss_fn 
    def construct(self, batch, n_particles, obs_count, condition_values):
        idx, X, D, library_size = batch
        print(condition_values.keys())
        if condition_values is None:
            condition_values = dict()
        guide_dists, guide_samples = self.guide(X,D,condition_values,n_particles)
        for k, v in guide_samples.items():
            condition_values[k] = v
        print(condition_values.keys())
        model_dists, model_samples = self.model(D,condition_values,n_particles)
        loss, metrics = self.loss_fn(guide_dists, model_dists, model_samples, D, X, obs_count)
        return loss
