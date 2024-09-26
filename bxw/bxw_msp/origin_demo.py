import pandas as pd
from typing import Any, Dict, List, Literal, Iterable, Optional, Sequence, Tuple, Union, Mapping, TypedDict
import anndata
import numpy as np
from tqdm.auto import tqdm
from mindspore.dataset import GeneratorDataset
from sklearn.model_selection import train_test_split
import mindspore
from mindspore import context, ops, Parameter, Tensor
import mindspore.numpy as mnp
from mindspore.nn.probability.distribution import Normal, Distribution, Bernoulli
import mindspore.nn as nn
import mindspore.nn.probability.distribution as msd
from mindspore.ops import broadcast_to, log as ms_log
import mindspore.dataset as ds
import scipy as sp
from mindspore.train.callback import LossMonitor
from mindspore.ops import BroadcastTo
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype

# context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

LIKELIHOOD_KEY_DTYPE = Literal["normal", "library_nb"]  # 只能是正态分布、泊松分布、负二项分布
RESULTS_BASE_DIR = "results/"
STEP_OUTPUT = Optional[Union[mindspore.Tensor, Mapping[str, Any]]]
config = {
        "name": "sams_vae_replogle",
        "seed": 0,
        "max_epochs": 1000,
        "data_module": "ReplogleDataModule",
        "data_module_kwargs": {"batch_size": 128},
        "model": "SAMSVAEModel",
        "model_kwargs": {
            "n": 128,
            "n_latent": 100,
            "mask_prior_prob": 0.01,
            "embedding_prior_scale": 1,
            "likelihood_key": "library_nb",
            "decoder_n_layers": 1,
            "decoder_n_hidden": 350
        },
        "guide": "SAMSVAEMeanFieldNormalGuide",
        "guide_kwargs": {
            "n_latent": 100,
            "basal_encoder_n_layers": 1,
            "basal_encoder_n_hidden": 350,
            "basal_encoder_input_normalization": "log_standardize"
        },
        "loss_module": "SAMSVAE_ELBOLossModule",
        "lightning_module_kwargs": {
            "lr": 0.001,
            "n_particles": 5
        },
        "predictor": "SAMSVAEPredictor"
    }

def _log_prob(value, mean=None, sd=None):
        """
        Evaluate log probability.

        Args:
            value (Tensor): The value to be evaluated.
            mean (Tensor): The mean of the distribution. Default: self.mean.
            sd (Tensor): The standard deviation of the distribution. Default: self.sd.

        Returns:
            Tensor: The log probability of the value.
        """

        # Compute the log probability
        unnormalized_log_prob = -0.5 * (ops.sqrt((value - mean) / sd))
        coff = -0.5 * ops.log(2 * Tensor(np.pi))
        neg_normalization = coff - ops.log(sd)
        return unnormalized_log_prob + neg_normalization

class RelaxedBernoulli(nn.Cell):
    def __init__(self, temperature, probs=None, logits=None):
        super(RelaxedBernoulli, self).__init__()
        if probs is not None:
            self.logits = ops.log(probs) - ops.log(1 - probs)
        else:
            self.logits = logits
        self.temperature = temperature

    def sample(self, sample_shape=()):
        print("===RelaxedBernoulli===")
        print(sample_shape)
        shape = sample_shape + self.logits.shape
        print(shape)
        minval = mindspore.Tensor(0, dtype=mindspore.float32) 
        maxval = mindspore.Tensor(1, dtype=mindspore.float32)
        uniform = ops.uniform(shape, minval, maxval)
        gumbel = -ops.log(-ops.log(uniform))
        y = (self.logits + gumbel) / self.temperature
        return ops.sigmoid(y)
    
    def log_prob(self, value):
        log_p = ops.log(value) * (self.logits - ops.log(self.temperature)) \
                - ops.log(1 + ops.exp(self.logits - ops.log(self.temperature)))
        return log_p


class RelaxedBernoulliStraightThrough(RelaxedBernoulli):
    def __init__(self, temperature, probs=None, logits=None):
        super(RelaxedBernoulliStraightThrough, self).__init__(temperature, probs, logits)

    def rsample(self, sample_shape=()):
        # Soft sample from the RelaxedBernoulli distribution
        soft_sample = super(RelaxedBernoulliStraightThrough, self).sample(sample_shape)
        # Clamp the probabilities to avoid numerical issues
        soft_sample = ops.clip_by_value(soft_sample, 1e-6, 1 - 1e-6)
        # Quantize the soft sample to get a hard sample
        hard_sample = ops.round(soft_sample)
        # Straight-through estimator
        hard_sample = ops.stop_gradient(hard_sample - soft_sample) + soft_sample
        return hard_sample

    def log_prob(self, value):
        # During backprop, we use the unquantized sample
        return super(RelaxedBernoulliStraightThrough, self).log_prob(value)


class GumbelSoftmaxBernoulliStraightThrough(RelaxedBernoulliStraightThrough):
    def __init__(self, temperature, probs=None, logits=None):
        super(GumbelSoftmaxBernoulliStraightThrough, self).__init__(temperature, probs, logits)
        self.probs = ops.sigmoid(self.logits)

    def log_prob(self, value):
        if self.probs is None:
            raise ValueError("probs must be defined to use this method.")
        return Bernoulli(probs=self.probs).log_prob(value)

    @property
    def mode(self):
        if self.probs is None:
            raise ValueError("probs must be defined to use this method.")
        mode = (self.probs > 0.5).astype(self.probs.dtype)
        return mode


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
        self.n = n
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
        self.generative_dists = {
            "p_z_basal": Normal(ops.zeros((self.n, self.n_latent), mindspore.float32), ops.ones((self.n, self.n_latent), mindspore.float32)),
            "p_E": Normal(self.p_E_loc, self.p_E_scale),
            "p_mask": Bernoulli(probs=ops.sigmoid(ops.log(self.p_mask_probs)))
        }
    def get_var_keys(self) -> List[str]:
        return ["z_basal", "E", "mask"]

    def construct(
        self,
        D,           
        condition_values,
        n_particles,       
    ) -> Tuple[Dict[str, Distribution], Dict[str, mindspore.Tensor]]:
        if condition_values is None:                        # 如果没有提供condition_value，则初始化为空字典
            condition_values = dict()
        
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
        z = samples["z_basal"] + ops.matmul(D, samples["E"] * samples["mask"])
        if self.likelihood_key != "library_nb":
            self.generative_dists["p_x"] = self.decoder(z)
        else:
            self.generative_dists["p_x"] = self.decoder(z, condition_values["library_size"])
        samples["x"] = self.generative_dists["p_x"].sample()
        return self.generative_dists, samples
    

class SAMSVAE_ELBOLossModule(nn.Cell):
    def __init__(
        self,
        model: nn.Cell,
        guide: nn.Cell,
        local_variables=["z_basal"],
        perturbation_plated_variables=["E", "mask"],
    ):
        super().__init__()
        local_variables = list(local_variables) if local_variables is not None else []
        perturbation_plated_variables = (
            list(perturbation_plated_variables)
            if perturbation_plated_variables is not None
            else []
        )

        assert sorted(model.get_var_keys()) == sorted(
            guide.get_var_keys()
        ), "Mismatch in model and guide variables"
        variables = local_variables + perturbation_plated_variables
        assert sorted(list(model.get_var_keys())) == sorted(
            variables
        ), "Mismatch between model variables and variables specified to loss module"
        self.model = model
        self.guide = guide
        self.local_variables = local_variables
        self.perturbation_plated_variables = perturbation_plated_variables

    def construct(
        self,
        X,
        D,
        condition_values,
        D_obs_counts,
        n_particles,
    ):
        print("进入LossModule")
        if condition_values is None:
            condition_values = dict()
        print("进入Guide")
        guide_dists, guide_samples = self.guide(
            X=X,
            D=D,
            condition_values=condition_values,
            n_particles=n_particles,
        )
        print("完成Guide")
        for k, v in guide_samples.items():
            condition_values[k] = v
        print("进入Model")
        model_dists, model_samples = self.model(
            D=D,
            condition_values=condition_values,
            n_particles=n_particles,
        )
        print("完成Model")
        # return guide_dists, model_dists, model_samples

        loss_terms = {}
        loss_terms["reconstruction"] = model_dists["p_x"].log_prob(X).sum(-1)
        for k in guide_dists.keys():
            var_key = k[2:]  # drop 'q_'(去掉前缀q_，得到Z_basal、E、mask)
            if var_key == "z_basal":
                loss_term = model_dists[f"p_{var_key}"].log_prob(model_samples[var_key])
                q_dist_dict = guide_dists[f"q_{var_key}"]
                sample, means, stds = q_dist_dict['sample'], q_dist_dict['means'], q_dist_dict['stds']
                loss_term = loss_term - _log_prob(sample, means, stds)
            else:
                loss_term = model_dists[f"p_{var_key}"].log_prob(model_samples[var_key])
                loss_term = loss_term - guide_dists[f"q_{var_key}"].log_prob(
                    model_samples[var_key]
                )
            if var_key in self.perturbation_plated_variables:   # (如果是E、mask，则重加权)
                loss_term = self._compute_reweighted_perturbation_plated_loss_term(
                    D, D_obs_counts, loss_term
                )
            loss_term = loss_term.sum(-1)
            loss_terms[var_key] = loss_term
        batch_elbo: mindspore.Tensor = sum([v for k, v in loss_terms.items()])
        loss = -batch_elbo.mean()
        metrics = {
            f"loss_term_{k}": -v.mean() for k, v in loss_terms.items()
        }
        return loss, metrics

    def loss(
        self,
        X: mindspore.Tensor,
        D: mindspore.Tensor,
        D_obs_counts: mindspore.Tensor,
        condition_values: Optional[Dict[str, mindspore.Tensor]] = None,
        n_particles: int = 1,
    ):
        guide_dists, model_dists, samples = self.construct(
            X=X,
            D=D,
            condition_values=condition_values,
            n_particles=n_particles,
        )
        loss_terms = {}
        loss_terms["reconstruction"] = model_dists["p_x"].log_prob(X).sum(-1)
        for k in guide_dists.keys():
            var_key = k[2:]  # drop 'q_'(去掉前缀q_，得到Z_basal、E、mask)
            if var_key == "z_basal":
                loss_term = model_dists[f"p_{var_key}"].log_prob(samples[var_key])
                q_dist_dict = guide_dists[f"q_{var_key}"]
                sample, means, stds = q_dist_dict['sample'], q_dist_dict['means'], q_dist_dict['stds']
                loss_term = loss_term - _log_prob(sample, means, stds)
            else:
                loss_term = model_dists[f"p_{var_key}"].log_prob(samples[var_key])
                loss_term = loss_term - guide_dists[f"q_{var_key}"].log_prob(
                    samples[var_key]
                )
            if var_key in self.perturbation_plated_variables:   # (如果是E、mask，则重加权)
                loss_term = self._compute_reweighted_perturbation_plated_loss_term(
                    D, D_obs_counts, loss_term
                )
            loss_term = loss_term.sum(-1)
            loss_terms[var_key] = loss_term
        batch_elbo: mindspore.Tensor = sum([v for k, v in loss_terms.items()])
        loss = -batch_elbo.mean()
        metrics = {
            f"loss_term_{k}": -v.mean() for k, v in loss_terms.items()
        }
        return loss, metrics

    def _compute_reweighted_perturbation_plated_loss_term(
        self, conditioning_variable, total_obs_per_condition, loss_term
    ):
        condition_nonzero = (conditioning_variable != 0).type(mindspore.float32)
        obs_scaling = 1 / total_obs_per_condition
        obs_scaling[ops.isinf(Tensor(obs_scaling,mindspore.float64))] = 0
        obs_scaling = obs_scaling.reshape(1, -1)
        rw_condition_nonzero = condition_nonzero * obs_scaling
        rw_loss_term = ops.matmul(rw_condition_nonzero, loss_term)
        return rw_loss_term
    

class PerturbationDataSample(TypedDict):
    idx: int
    X: mindspore.Tensor
    D: mindspore.Tensor


class PerturbationDataset(ds.Dataset):
    def __getitem__(self, idx: int) -> PerturbationDataSample:
        raise NotImplementedError

    def get_dosage_obs_per_dim(self) -> mindspore.Tensor:
        raise NotImplementedError

    def convert_idx_to_ids(self, idx: np.array) -> np.array:
        raise NotImplementedError

# class TensorPerturbationDataset(PerturbationDataset):
#     def __init__(
#         self,
#         X: mindspore.Tensor,
#         D: mindspore.Tensor,
#         ids: Optional[Iterable] = None,
#     ):
#         super().__init__() 
#         self.parent = None
#         self.X = X  # observations
#         self.D = D  # perturbations
#         if ids is None:
#             self.ids = np.arange(len(X))
#         else:
#             self.ids = np.array(ids)

#         self.D_obs_per_dim = (self.D != 0).sum(0)

#         self.library_size = self.X.sum(1)

#     def __getitem__(self, idx: int) -> PerturbationDataSample:
#         return dict(idx=idx, X=self.X[idx], D=self.D[idx])

#     def __len__(self):
#         return len(self.X)

#     def get_dosage_obs_per_dim(self):
#         return self.D_obs_per_dim

#     def convert_idx_to_ids(self, idx: np.array) -> np.array:
#         return self.ids[idx]



def estimate_data_average_treatment_effects(
    adata: anndata.AnnData,
    label_col: str,
    control_label: Any,
    method: Literal["mean", "perturbseq"],
    compute_fdr: bool = False,
) -> anndata.AnnData:
    if compute_fdr:
        raise NotImplementedError
    valid_methods = ["mean", "perturbseq"]
    assert method in valid_methods, f"Method must be one of {valid_methods}"
    perturbations = adata.obs[label_col].unique()
    assert control_label in perturbations
    alt_labels = [x for x in perturbations if x != control_label]
    X_control = adata[adata.obs[label_col] == control_label].X
    if sp.sparse.issparse(X_control):
        X_control = X_control.toarray()
    if method == "perturbseq":
        X_control = 1e4 * X_control / np.sum(X_control, axis=1, keepdims=True)
        X_control = np.log2(X_control + 1)
    X_control_mean = X_control.mean(0)
    average_effects = []
    for alt_label in tqdm(alt_labels):
        X_alt = adata[adata.obs[label_col] == alt_label].X
        if sp.sparse.issparse(X_alt):
            X_alt = X_alt.toarray()
        if method == "perturbseq":
            X_alt = 1e4 * X_alt / np.sum(X_alt, axis=1, keepdims=True)
            X_alt = np.log2(X_alt + 1)
        X_alt_mean = X_alt.mean(0)
        average_effects.append(X_alt_mean - X_control_mean)
    average_effects = np.stack(average_effects)
    results = anndata.AnnData(
        obs=pd.DataFrame(index=alt_labels),
        X=average_effects,
        var=adata.var.copy(),
        uns=dict(control=control_label),
    )
    return results


class ObservationNormalizationStatistics:
    def __init__(self):
        self.x_mean = None
        self.x_std = None
        self.log_x_mean = None
        self.log_x_std = None

    def set_statistics(self, x_mean, x_std, log_x_mean, log_x_std):
        self.x_mean = x_mean
        self.x_std = x_std
        self.log_x_mean = log_x_mean
        self.log_x_std = log_x_std


class PerturbationDataModule:
    def get_train_perturbation_obs_counts(self) -> mindspore.Tensor:
        raise NotImplementedError

    def get_val_perturbation_obs_counts(self) -> mindspore.Tensor:
        raise NotImplementedError

    def get_test_perturbation_obs_counts(self) -> mindspore.Tensor:
        raise NotImplementedError

    def get_x_var_info(self) -> pd.DataFrame:
        raise NotImplementedError

    def get_d_var_info(self) -> pd.DataFrame:
        raise NotImplementedError

    def get_obs_info(self) -> pd.DataFrame:
        raise NotImplementedError

    def get_x_train_statistics(self) -> ObservationNormalizationStatistics:
        raise NotImplementedError

    def get_unique_observed_intervention_info(self) -> pd.DataFrame:
        raise NotImplementedError

    def get_unique_observed_intervention_dosages(
        self, pert_names: Sequence
    ) -> mindspore.Tensor:
        raise NotImplementedError

    def get_estimated_average_treatment_effects(
        self,
        method: Literal["mean", "perturbseq"],
        split: Optional[str] = None,
    ) -> Optional[anndata.AnnData]:
        return None

    def get_simulated_latent_effects(self) -> Optional[anndata.AnnData]:
        return None


def data_generator(X: mindspore.Tensor, D: mindspore.Tensor, ids: Iterable, id_to_index_map: dict):
        library_size = X.sum(1)
        for idx in ids:
            index = id_to_index_map[idx]
            yield index, X[index].asnumpy(), D[index].asnumpy(), library_size[index].asnumpy()


def get_dosage_obs_per_dim(D: mindspore.Tensor):
        D_obs_per_dim = (D != 0).sum(0)
        return D_obs_per_dim


class ReplogleDataModule(PerturbationDataModule):
    def __init__(
        self,
        batch_size: int = 128,
        data_path: Optional[str] = "/home/ma-user/work/sams_vae/dataset/replogle.h5ad",
    ):
        super().__init__()
        self.batch_size = batch_size
        self.adata = anndata.read_h5ad(data_path)
        idx = np.arange(self.adata.shape[0])
        train_idx, test_idx = train_test_split(idx, train_size=0.8, random_state=0)
        train_idx, val_idx = train_test_split(train_idx, train_size=0.8, random_state=0)
        self.adata.obs["split"] = None
        self.adata.obs.iloc[
            train_idx, self.adata.obs.columns.get_loc("split")
        ] = "train"
        self.adata.obs.iloc[val_idx, self.adata.obs.columns.get_loc("split")] = "val"
        self.adata.obs.iloc[test_idx, self.adata.obs.columns.get_loc("split")] = "test"
        self.adata.obs["T"] = self.adata.obs["sgID_AB"].apply(
            lambda x: "non-targeting" if "non-targeting" in x else x
        )
        dosage_df = pd.get_dummies(self.adata.obs["T"])
        dosage_df = dosage_df.drop(columns=["non-targeting"])
        self.d_var_info = dosage_df.T[[]]
        dosage_array = dosage_df.astype(np.float32).values
        D = mindspore.Tensor(dosage_array, mindspore.float32)
        # D = mindspore.Tensor(dosage_df.astype(np.float32))
        X = mindspore.Tensor(self.adata.X.copy())
        id_to_index_map = {id_: i for i, id_ in enumerate(self.adata.obs.index)}
        self.ids_tr = self.adata.obs[self.adata.obs["split"] == "train"].index
        # X_tr = X[(self.adata.obs["split"] == "train").to_numpy()]
        # D_tr = D[(self.adata.obs["split"] == "train").to_numpy()]
        self.ids_val = self.adata.obs[self.adata.obs["split"] == "val"].index
        # X_val = X[(self.adata.obs["split"] == "val").to_numpy()]
        # D_val = D[(self.adata.obs["split"] == "val").to_numpy()]
        self.ids_test = self.adata.obs[self.adata.obs["split"] == "test"].index
        # X_test = X[(self.adata.obs["split"] == "test").to_numpy()]
        # D_test = D[(self.adata.obs["split"] == "test").to_numpy()]
        train_indices = mindspore.Tensor((self.adata.obs["split"] == "train").to_numpy(), mindspore.bool_)
        val_indices = mindspore.Tensor((self.adata.obs["split"] == "val").to_numpy(), mindspore.bool_)
        test_indices = mindspore.Tensor((self.adata.obs["split"] == "test").to_numpy(), mindspore.bool_)

        self.X_tr = X[train_indices]
        self.D_tr = D[train_indices]
        self.X_val = X[val_indices]
        self.D_val = D[val_indices]
        self.X_test = X[test_indices]
        self.D_test = D[test_indices]
        self.train_dataset = data_generator(self.X_tr, self.D_tr, self.ids_tr, id_to_index_map) 
        self.val_dataset = data_generator(self.X_val, self.D_val, self.ids_val, id_to_index_map) 
        self.test_dataset = data_generator(self.X_test, self.D_test, self.ids_test, id_to_index_map)
        # self.train_dataset = SCRNASeqTensorPerturbationDataset(
        #     X=X_tr, D=D_tr, ids=ids_tr
        # )
        # self.val_dataset = SCRNASeqTensorPerturbationDataset(
        #     X=X_val, D=D_val, ids=ids_val
        # )
        # self.test_dataset = SCRNASeqTensorPerturbationDataset(
        #     X=X_test, D=D_test, ids=ids_test
        # )
        self.x_train_statistics = ObservationNormalizationStatistics()
        x_tr_mean = self.X_tr.mean(0)
        x_tr_std = self.X_tr.std(0)
        log_x_tr = ops.log(self.X_tr + 1)
        log_x_tr_mean = log_x_tr.mean(0)
        log_x_tr_std = log_x_tr.std(0)
        self.x_train_statistics.set_statistics(
            x_mean=x_tr_mean, 
            x_std=x_tr_std, 
            log_x_mean=log_x_tr_mean, 
            log_x_std=log_x_tr_std, 
        )
        df = self.adata.obs.groupby("T")["split"].agg(set).reset_index()
        for split in ["train", "val", "test"]:
            df[split] = df["split"].apply(lambda x: split in x)
        df = df.set_index("T").drop(columns=["split"])
        self.unique_observed_intervention_df = df
        self.adata.obs["i"] = np.arange(self.adata.shape[0])
        idx_map = self.adata.obs.drop_duplicates("T").set_index("T")["i"].to_dict()
        self.unique_intervention_dosage_map = {k: D[v] for k, v in idx_map.items()}

    # def train_dataloader(self): 
    #     dataset = ds.GeneratorDataset(self.train_dataset, column_names=["X", "D", "ids"]) 
    #     dataset = dataset.shuffle(buffer_size=len(self.train_dataset))
    #     dataset = dataset.batch(batch_size=self.batch_size, drop_remainder=True) 
    #     return dataset

    def train_dataloader(self): 
        dataset = ds.GeneratorDataset(
            source=self.train_dataset,
            column_names=["idx", "X", "D", "library_size"]
        )
        dataset = dataset.batch(batch_size=self.batch_size, drop_remainder=True)
        return dataset

    def val_dataloader(self): 
        dataset = ds.GeneratorDataset(
            source=self.val_dataset,
            column_names=["idx", "X", "D", "library_size"]
        )
        dataset = dataset.batch(batch_size=self.batch_size, drop_remainder=True)
        return dataset
    
    def test_dataloader(self): 
        dataset = ds.GeneratorDataset(
            source=self.test_dataset,
            column_names=["idx", "X", "D", "library_size"]
        )
        dataset = dataset.batch(batch_size=self.batch_size, drop_remainder=True)
        return dataset

    def get_train_perturbation_obs_counts(self) -> mindspore.Tensor:
        return get_dosage_obs_per_dim(self.D_tr)

    def get_val_perturbation_obs_counts(self) -> mindspore.Tensor:
        return get_dosage_obs_per_dim(self.D_val)

    def get_test_perturbation_obs_counts(self) -> mindspore.Tensor:
        return get_dosage_obs_per_dim(self.D_test)

    def get_x_var_info(self) -> pd.DataFrame:
        return self.adata.var.copy()

    def get_d_var_info(self) -> pd.DataFrame:
        return self.d_var_info.copy()

    def get_obs_info(self) -> pd.DataFrame:
        return self.adata.obs.copy()

    def get_x_train_statistics(self) -> ObservationNormalizationStatistics:
        return self.x_train_statistics

    def get_unique_observed_intervention_info(self) -> pd.DataFrame:
        return self.unique_observed_intervention_df.copy()

    def get_unique_observed_intervention_dosages(
        self, pert_names: Sequence
    ) -> mindspore.Tensor:
        D = ops.zeros((len(pert_names), self.d_var_info.shape[0]))
        for i, pert_name in enumerate(pert_names):
            D[i] = self.unique_intervention_dosage_map[pert_name]
        return D

    def get_estimated_average_treatment_effects(
        self,
        method: Literal["mean", "perturbseq"],
        split: Optional[str] = None,
    ) -> Optional[anndata.AnnData]:
        adata = self.adata
        if split is not None:
            adata = adata[adata.obs["split"] == split]
        return estimate_data_average_treatment_effects(
            adata,
            label_col="T",
            control_label="non-targeting",
            method=method,
        )

    def get_simulated_latent_effects(self) -> Optional[anndata.AnnData]:
        return None
    

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

    def construct(self, x: mindspore.Tensor):
        multiple_particles = len(x.shape) == 3
        if multiple_particles:
            n_particles, n, x_dim = x.shape
            x = x.reshape(n_particles * n, x_dim)
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
        eps = ops.StandardNormal()(means.shape)
        sample = means + stds * eps
        dist_dict = {
            "sample": sample,
            "means": means,
            "stds": stds,
            "log_var": log_var,
            "vars": vars,
        }
        return dist_dict


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


class BaseLibraryGammaPoissonSharedConcentrationLikelihood(nn.Cell):
    def construct(self, x: mindspore.Tensor, library_size: mindspore.Tensor):
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
        concentration = ops.stop_gradient(concentration) 
        mu = ops.stop_gradient(mu)
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

    
class GammaPoisson(nn.Cell):
    """
    Compound distribution comprising of a gamma-poisson pair, also referred to as
    a gamma-poisson mixture. The `rate` parameter for the Poisson distribution
    is unknown and randomly drawn from a Gamma distribution.
    
    :param concentration: Shape parameter (alpha) of the Gamma distribution.
    :param rate: Rate parameter (beta) for the Gamma distribution.
    """

    def __init__(self, concentration, rate, validate_args=None):
        super(GammaPoisson, self).__init__()
        concentration, rate = self._broadcast_all(concentration, rate)
        self._gamma = msd.Gamma(concentration, rate)
        self._validate_args = validate_args

    @staticmethod
    def _broadcast_all(*args):
        shape = np.broadcast_shapes(*[arg.shape for arg in args])
        return tuple(BroadcastTo(shape)(arg) for arg in args)

    @property
    def concentration(self):
        return self._gamma.concentration

    @property
    def rate(self):
        return self._gamma.rate

    def sample(self, sample_shape=()):
        rate_sample = self._gamma.sample(sample_shape)
        poisson_dist = msd.Poisson(rate_sample)
        return poisson_dist.sample()

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        post_value = self.concentration + value
        log_beta = ops.Lgamma()(self.concentration) + ops.Lgamma()(value + 1) - ops.Lgamma()(post_value)
        return (
            -log_beta
            - P.Log()(post_value)
            + P.Log()(self.concentration) * P.Log()(self.rate)
            - P.Log()(post_value) * P.Log()(1 + self.rate)
        )

    @property
    def mean(self):
        return self.concentration / self.rate

    @property
    def variance(self):
        return self.concentration / (self.rate ** 2) * (1 + self.rate)

    def expand(self, batch_shape):
        new = GammaPoisson(
            broadcast_to(self.concentration, batch_shape),
            broadcast_to(self.rate, batch_shape),
        )
        return new

    def _validate_sample(self, value):
        if not mindspore.numpy.all(value >= 0):
            raise ValueError("Sample values must be non-negative")


def estimate_model_average_treatment_effect(
    model: nn.Cell,
    guide: nn.Cell,
    dosages_alt: mindspore.Tensor,
    dosages_control: mindspore.Tensor,
    n_particles: int,
    method: Literal["mean", "perturbseq"],
    condition_values: Optional[Dict[str, mindspore.Tensor]] = None,
    batch_size: int = 500,
    dosage_independent_variables: Optional[List[str]] = None,
    seed: int = 0,
):
    mindspore.manual_seed(seed)
    valid_methods = ["mean", "perturbseq"]
    assert method in valid_methods, f"Method must be one of {valid_methods}"
    if condition_values is None:
        condition_values = dict()
    device = next(model.parameters()).device
    for k, v in condition_values.items():
        if len(v.shape) == 2:
            condition_values[k] = v.unsqueeze(0).expand((batch_size, -1, -1))
        condition_values[k] = v.to(device)
    curr_condition_values = {k: v[:1] for k, v in condition_values.items()}
    _, model_samples = model(
        D=dosages_control, condition_values=curr_condition_values, n_particles=1
    )
    n_phenos = model_samples["x"].shape[-1]
    X_control_sum = ops.zeros((1, n_phenos))
    X_alt_sums = ops.zeros((dosages_alt.shape[0], n_phenos))
    for i in range(0, n_particles, batch_size):
        curr_num_particles = min(batch_size, n_particles - i)
        curr_condition_values = {}
        for k, v in condition_values.items():
            if v.shape[0] == n_particles:
                curr_condition_values[k] = v[i : i + curr_num_particles].to(device)
            else:
                curr_condition_values[k] = v.to(device)
        guide_dists, guide_samples = guide(
            n_particles=curr_num_particles, condition_values=curr_condition_values
        )
        for k, v in guide_samples.items():
            curr_condition_values[k] = v
        _, model_samples = model(
            D=dosages_control,
            condition_values=curr_condition_values,
            n_particles=curr_num_particles,
        )
        if dosage_independent_variables is not None:
            for k in dosage_independent_variables:
                curr_condition_values[k] = model_samples[k]
        X_control = model_samples["x"].squeeze(1)
        if method == "perturbseq":
            X_control = 1e4 * X_control / ops.sum(X_control, dim=1, keepdim=True)
            X_control = ops.log2(X_control + 1)
        X_control_sum[0] += X_control.sum(0)
        for t_idx in tqdm(range(dosages_alt.shape[0])):
            D_curr = dosages_alt[t_idx : t_idx + 1]
            _, model_samples = model(
                D=D_curr,
                condition_values=curr_condition_values,
                n_particles=curr_num_particles,
            )
            X_curr = model_samples["x"].squeeze(1)
            if method == "perturbseq":
                # standardize by library size and log normalize
                X_curr = 1e4 * X_curr / ops.sum(X_curr, dim=1, keepdim=True)
                X_curr = ops.log2(X_curr + 1)
            X_alt_sums[t_idx] += X_curr.sum(0)
    X_control_mean = X_control_sum / n_particles
    X_alt_means = X_alt_sums / n_particles
    average_effects = (X_alt_means - X_control_mean).detach().cpu().numpy()
    return average_effects


class SAMSVAEPredictor(nn.Cell):
    def __init__(
        self,
        model: nn.Cell,
        guide: nn.Cell,
        local_variables=["z_basal"],       # 一个可选的可迭代字符串列表，表示局部变量名称
        perturbation_plated_variables=["E", "mask"],  # 一个可选的可迭代字符串列表，表示板阵化扰动变量名称
        dosage_independent_variables: Optional[Iterable[str]] = None,   # 一个可选的可迭代字符串列表，表示与剂量无关的变量名称
    ):
        super().__init__()

        local_variables = list(local_variables) if local_variables is not None else []
        perturbation_plated_variables = (
            list(perturbation_plated_variables)
            if perturbation_plated_variables is not None
            else []
        )
        assert sorted(model.get_var_keys()) == sorted(
            guide.get_var_keys()
        ), "Mismatch in model and guide variables"
        variables = local_variables + perturbation_plated_variables
        assert sorted(list(model.get_var_keys())) == sorted(
            variables
        ), "Mismatch between model variables and variables specified to loss module"
        if dosage_independent_variables is not None:
            assert set(dosage_independent_variables).issubset(set(variables))
        self.model = model.set_train(False)                  # 将传入的model设置为评估模式
        self.guide = guide.set_train(False)                   # 将传入的guide设置为评估模式
        self.local_variables = local_variables
        self.perturbation_plated_variables = perturbation_plated_variables
        self.dosage_independent_variables = dosage_independent_variables

    def _get_device(self):
        device = next(self.model.parameters()).device
        return device

    def compute_predictive_iwelbo(
        self,
        loaders: Union[GeneratorDataset, Sequence[GeneratorDataset]],  # 一个GeneratorDataset或GeneratorDataset序列，包含用于计算IWELBO的perturbation数据集
        n_particles: int,                                  # 用于计算预测IWELBO的粒子数量
    ) -> pd.DataFrame:
        if isinstance(loaders, GeneratorDataset):
            loaders = [loaders]
        device = self._get_device()
        guide_dists, guide_samples = self.guide(n_particles=n_particles)
        condition_values = {}
        for var_name in self.perturbation_plated_variables:
            condition_values[var_name] = guide_samples[var_name]
        id_list = []                        # 用于存储样本ID
        iwelbo_list = []                    # 用于存储计算得到的IWELBO值
        for loader in loaders:
            idx_list_curr = []
            for batch in tqdm(loader):
                for k in batch:
                    batch[k] = batch[k].to(device)
                idx_list_curr.append(batch["idx"].detach().cpu().numpy())
                if self.model.likelihood_key == "library_nb":
                    condition_values["library_size"] = batch["library_size"]
                guide_dists, guide_samples = self.guide(
                    X=batch["X"],
                    D=batch["D"],
                    condition_values=condition_values,
                    n_particles=n_particles,
                )
                if self.model.likelihood_key == "library_nb":
                    guide_samples["library_size"] = batch["library_size"]
                model_dists, model_samples = self.model(
                    D=batch["D"],
                    condition_values=guide_samples,
                    n_particles=n_particles,
                )
                iwelbo_terms_dict = {}  # 初始化一个空字典iwelbo_terms_dict，用于存储每种变量的IWELBO项
                iwelbo_terms_dict["x"] = model_dists["p_x"].log_prob(batch["X"]).sum(-1)
                for var_name in self.local_variables:
                    p = (
                        model_dists[f"p_{var_name}"]
                        .log_prob(guide_samples[var_name])
                        .sum(-1)
                    )
                    q = (
                        guide_dists[f"q_{var_name}"]
                        .log_prob(guide_samples[var_name])
                        .sum(-1)
                    )
                    iwelbo_terms_dict[var_name] = p - q
                iwelbo_terms = sum([v for k, v in iwelbo_terms_dict.items()])
                batch_iwelbo = ops.logsumexp(iwelbo_terms, axis=0) - np.log(
                    n_particles
                )
                iwelbo_list.append(batch_iwelbo.detach().cpu().numpy())

            idx_curr = np.concatenate(idx_list_curr)
            dataset: PerturbationDataset = loader.dataset
            ids_curr = dataset.convert_idx_to_ids(idx_curr)
            id_list.append(ids_curr)
 
        iwelbo = np.concatenate(iwelbo_list)           # 将所有批次的IWELBO值合并成一个数组iwelbo
        ids = np.concatenate(id_list)                  # 将所有批次的样本ID合并成一个数组ids

        iwelbo_df = pd.DataFrame(
            index=ids, columns=["IWELBO"], data=iwelbo.reshape(-1, 1)
        )
        return iwelbo_df

    def sample_observations(
        self,
        dosages: mindspore.Tensor,                                           # 感兴趣的扰动的编码剂量
        perturbation_names: Optional[Sequence[str]],                     # 每个扰动的名称
        n_particles: int = 1,                                            # 每个剂量采样的数量，默认为1
        condition_values: Optional[Dict[str, mindspore.Tensor]] = None,      # 额外的条件变量，其键是变量名，值是 mindspore.Tensor变量
        x_var_info: Optional[pd.DataFrame] = None,                       # 观测变量x的信息
    ) -> anndata.AnnData:
        device = self._get_device()
        dosages = dosages.to(device)
        guide_dists, guide_samples = self.guide(n_particles=n_particles)
        if condition_values is None:
            condition_values = dict()
        else:
            condition_values = {k: v.to(device) for k, v in condition_values.items()}
        for var_name in self.perturbation_plated_variables:
            condition_values[var_name] = guide_samples[var_name]

        x_samples_list = []   # 初始化一个空列表x_samples_list，用于存储每次采样得到的观测数据
        for i in tqdm(range(dosages.shape[0])):
            D = dosages[i : i + 1]  # 从dosages张量中提取当前索引对应的扰动剂量
            _, model_samples = self.model(
                D=D, condition_values=condition_values, n_particles=n_particles
            )
            x_samples_list.append(model_samples["x"].detach().cpu().numpy().squeeze())

        x_samples = np.concatenate(x_samples_list) 
        obs = pd.DataFrame(index=np.arange(x_samples.shape[0])) 
        obs["perturbation_idx"] = np.repeat(np.arange(dosages.shape[0]), n_particles)
        obs["particle_idx"] = np.tile(np.arange(dosages.shape[0]), n_particles)
        if perturbation_names is not None:
            obs["perturbation_name"] = np.array(perturbation_names)[
                obs["perturbation_idx"].to_numpy()
            ]

        adata = anndata.AnnData(obs=obs, X=x_samples)
        if x_var_info is not None:
            adata.var = x_var_info.copy()
        return adata

    def sample_observations_data_module(
        self,
        data_module: PerturbationDataModule,
        n_particles: int,
        condition_values: Optional[Dict[str, mindspore.Tensor]] = None,
    ):

        perturbation_names = data_module.get_unique_observed_intervention_info().index
        D = data_module.get_unique_observed_intervention_dosages(perturbation_names)
        x_var_info = data_module.get_x_var_info()

        adata = self.sample_observations(
            dosages=D,
            perturbation_names=perturbation_names,
            x_var_info=x_var_info,
            n_particles=n_particles,
            condition_values=condition_values,
        )

        return adata

    def estimate_average_treatment_effects(
        self,
        dosages_alt: mindspore.Tensor,                                      # 实验组的剂量张量
        dosages_control: mindspore.Tensor,                                  # 对照组的剂量张量
        method: Literal["mean", "perturbseq"],                          # 计算平均处理效应的方法，可以是 "mean" 或 "perturbseq"
        n_particles: int = 1000,                                        # 每个处理效应要采样的粒子数量，默认为1000
        condition_values: Optional[Dict[str, mindspore.Tensor]] = None,     # 额外的条件变量
        perturbation_names_alt: Optional[Sequence[str]] = None,         # 实验组剂量的名称
        perturbation_name_control: Optional[str] = None,                # 对照验组剂量的名称
        x_var_info: Optional[pd.DataFrame] = None,                      # 观测变量的信息
        batch_size: int = 500,                                          # 批量处理的大小，默认为500
    ) -> anndata.AnnData:

        device = self._get_device()
        dosages_alt = dosages_alt.to(device)
        dosages_control = dosages_control.to(device)
        if condition_values is not None:
            for k in condition_values:
                condition_values[k] = condition_values[k].to(device)

        average_treatment_effects = estimate_model_average_treatment_effect(
            model=self.model,
            guide=self.guide,
            dosages_alt=dosages_alt,
            dosages_control=dosages_control,
            n_particles=n_particles,
            method=method,
            condition_values=condition_values,
            batch_size=batch_size,
            dosage_independent_variables=self.dosage_independent_variables,
        )
        adata = anndata.AnnData(average_treatment_effects)
        if perturbation_names_alt is not None:
            adata.obs = pd.DataFrame(index=np.array(perturbation_names_alt))
        if perturbation_name_control is not None:
            adata.uns["control"] = perturbation_name_control
        if x_var_info is not None:
            adata.var = x_var_info.copy()
        return adata

    def estimate_average_effects_data_module(
        self,
        data_module: PerturbationDataModule,
        control_label: str,
        method: Literal["mean", "perturbseq"],
        n_particles: int = 1000,
        condition_values: Optional[Dict[str, mindspore.Tensor]] = None,
        batch_size: int = 500,
    ):
        perturbation_names = data_module.get_unique_observed_intervention_info().index
        perturbation_names_alt = [
            name for name in perturbation_names if name != control_label
        ]
        dosages_alt = data_module.get_unique_observed_intervention_dosages(
            perturbation_names_alt
        )
        dosages_ref = data_module.get_unique_observed_intervention_dosages(
            [control_label]
        )

        x_var_info = data_module.get_x_var_info()

        adata = self.estimate_average_treatment_effects(
            dosages_alt=dosages_alt,
            dosages_control=dosages_ref,
            method=method,
            n_particles=n_particles,
            condition_values=condition_values,
            perturbation_names_alt=perturbation_names_alt,
            perturbation_name_control=control_label,
            x_var_info=x_var_info,
            batch_size=batch_size,
        )
        return adata
    

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
        self.GumbelSoftmaxBernoull = GumbelSoftmaxBernoulliStraightThrough(temperature=self.gs_temperature,logits=self.q_mask_logits)
        self.q_E_dist = Normal(self.means, self.scale)
    def get_var_keys(self):
        var_keys = ["z_basal", "E", "mask"]
        return var_keys

    def construct(
        self,
        X: Optional[mindspore.Tensor] = None,
        D: Optional[mindspore.Tensor] = None,
        condition_values: Optional[Dict[str, mindspore.Tensor]] = None,
        n_particles: int = 1,
    ):  # object 是 MindSpore 分布类的占位符
        if condition_values is None:
            condition_values = dict()
        guide_distributions: Dict[str, object] = {}
        guide_samples: Dict[str, mindspore.Tensor] = {}
        guide_distributions["q_mask"] = self.GumbelSoftmaxBernoull
        # 计算 Normal 分布的标准差 
        guide_distributions["q_E"] = self.q_E_dist
        print("row 1344")
        if "mask" not in condition_values:
            guide_samples["mask"] = guide_distributions[f"q_mask"].rsample((n_particles,))
        else:
            guide_samples["mask"] = condition_values["mask"]
        if "E" not in condition_values:
            mean = guide_distributions["q_E"].mean()
            std = guide_distributions["q_E"].sd()
            epsilon = ops.standard_normal((n_particles, mean.shape[0], mean.shape[1]))  # 从标准正态分布采样
            print("epsilon:", epsilon.shape)
            guide_samples["E"] = mean + epsilon * std  # 重新参数化采样
        else:
            guide_samples["E"] = condition_values["E"]
        print("row 1356")
        if X is not None and D is not None:
            encoder_input = X
            if self.normalization_module is not None:
                encoder_input = self.normalization_module(encoder_input)
            
            encoder_shape = encoder_input.shape
            print(encoder_shape)

            # encoder_input = ops.expand_dims(encoder_input, 0)
            print("n_particles:",n_particles)
            encoder_input = ops.broadcast_to(encoder_input, (n_particles, encoder_shape[0], encoder_shape[1]))

            if not self.mean_field_encoder:
                latent_offset = ops.matmul(D, guide_samples["mask"] * guide_samples["E"])
                print(latent_offset.shape)
                print(encoder_input.shape)
                encoder_input = ops.concat([encoder_input, latent_offset], axis=-1)

            dict_q_z_basal = self.z_basal_encoder(encoder_input) # 参数化z_basal的变分分布
            guide_distributions["q_z_basal"] = dict_q_z_basal
            mean = dict_q_z_basal['means']
            std = dict_q_z_basal['stds']
            guide_samples["z_basal"] = dict_q_z_basal['sample']
        print("row 1375")
        if "z_basal" in condition_values:
            guide_samples["z_basal"] = condition_values["z_basal"]
        return guide_distributions, guide_samples
    

def add_data_info_to_config(config: Dict, data_module: PerturbationDataModule):
    if "model_kwargs" not in config:
        config["model_kwargs"] = dict()
    if "guide_kwargs" not in config:
        config["guide_kwargs"] = dict()
    config["model_kwargs"]["n_treatments"] = data_module.get_d_var_info().shape[0]
    config["model_kwargs"]["n_phenos"] = data_module.get_x_var_info().shape[0]
    config["guide_kwargs"]["n_treatments"] = data_module.get_d_var_info().shape[0]
    config["guide_kwargs"]["n_phenos"] = data_module.get_x_var_info().shape[0]
    config["guide_kwargs"][
        "x_normalization_stats"
    ] = data_module.get_x_train_statistics()
    return config


def train(config: Dict):
    data_module = ReplogleDataModule(batch_size=config["data_module_kwargs"]["batch_size"])
    config = add_data_info_to_config(config, data_module)

    # 设置随机种子
    mindspore.set_seed(config["seed"])

    # Step 1: 初始化模型、引导模块、损失模块和优化器
    model = SAMSVAEModel(
        n = config["model_kwargs"]["n"],
        n_latent=config["model_kwargs"]["n_latent"],
        n_treatments=config["model_kwargs"]["n_treatments"],
        n_phenos=config["model_kwargs"]["n_phenos"],
        mask_prior_prob=config["model_kwargs"]["mask_prior_prob"],
        embedding_prior_scale=config["model_kwargs"]["embedding_prior_scale"],
        likelihood_key=config["model_kwargs"]["likelihood_key"],
        decoder_n_layers=config["model_kwargs"]["decoder_n_layers"],
        decoder_n_hidden=config["model_kwargs"]["decoder_n_hidden"],
    )

    guide = SAMSVAEMeanFieldNormalGuide(
        n_latent=config["guide_kwargs"]["n_latent"],
        n_treatments=config["guide_kwargs"]["n_treatments"],
        n_phenos=config["guide_kwargs"]["n_phenos"],
        basal_encoder_n_layers=config["guide_kwargs"]["basal_encoder_n_layers"],
        basal_encoder_n_hidden=config["guide_kwargs"]["basal_encoder_n_hidden"],
        basal_encoder_input_normalization=config["guide_kwargs"]["basal_encoder_input_normalization"],
        x_normalization_stats=data_module.get_x_train_statistics()
    )

    loss_module = SAMSVAE_ELBOLossModule(model=model, guide=guide)
    # optimizer = nn.Adam(params=model.trainable_params(), learning_rate=config["lightning_module_kwargs"]["lr"])

    # 手动指定需要优化的参数
    params_to_pass = [
        loss_module.model.decoder.log_concentration,
        loss_module.model.decoder.mlp.layers[0][0].weight,
        loss_module.model.decoder.mlp.layers[0][0].bias,
        loss_module.model.decoder.normalized_mean_decoder[0].weight,
        loss_module.model.decoder.normalized_mean_decoder[0].bias,
        loss_module.guide.q_mask_logits,
        loss_module.guide.q_E_loc,
        loss_module.guide.q_E_log_scale,
        loss_module.guide.z_basal_encoder.mlp.layers[0][0].weight,
        loss_module.guide.z_basal_encoder.mlp.layers[0][0].bias,
        loss_module.guide.z_basal_encoder.mean_encoder.weight,
        loss_module.guide.z_basal_encoder.mean_encoder.bias,
        loss_module.guide.z_basal_encoder.log_var_encoder.weight,
        loss_module.guide.z_basal_encoder.log_var_encoder.bias
    ]


    # 将这些参数传给优化器
    optimizer = nn.Adam(params=params_to_pass, learning_rate=config["lightning_module_kwargs"]["lr"])

    def train_one_epoch(train_dataloader, model, guide, loss_module, optimizer, config):
        model.set_train()
        guide.set_train()
        total_loss = 0
        num_batches = 0

        for batch in train_dataloader:
            idx, X, D, library_size = batch  # 解包列表
            X = Tensor(X, mstype.float32)
            D = Tensor(D, mstype.float32)
            library_size = Tensor(library_size, mstype.float32)

            condition_values = {"idx": idx, "library_size": library_size}

            # loss, metrics = loss_module.loss(
            #     X, D, data_module.get_train_perturbation_obs_counts(),
            #     condition_values, n_particles=config["lightning_module_kwargs"]["n_particles"]
            # )
            D_obs_counts = data_module.get_train_perturbation_obs_counts()
            grad_fn = mindspore.value_and_grad(loss_module, weights=params_to_pass, has_aux=True)
            (loss, metrics), grad = grad_fn(X, D, condition_values, D_obs_counts, config["lightning_module_kwargs"]["n_particles"])
            
            # grads = ops.GradOperation(get_by_list=True)(loss_module, params_to_pass)(X, D, condition_values, config["lightning_module_kwargs"]["n_particles"])
                
            if not isinstance(grads, (list, tuple)): 
                grads = [grads]
            
            # 将梯度应用到优化器上进行参数更新
            # optimizer(grads)
            print("\nstep!!!!\n")
            total_loss += loss.asnumpy()
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"Average training loss: {avg_loss}")
        return avg_loss

    def validate(val_dataloader, model, guide, loss_module):
        model.set_train(False)
        val_loss = 0
        num_batches = 0

        for batch in val_dataloader:
            X = batch['X']
            D = batch['D']
            D_obs_counts = batch.get('library_size', None)
            
            with mindspore.no_grad():
                loss, _ = loss_module.loss(X, D, D_obs_counts, n_particles=config["lightning_module_kwargs"]["n_particles"])
            val_loss += loss.asnumpy()
            num_batches += 1
        
        avg_val_loss = val_loss / num_batches
        print(f"Validation loss: {avg_val_loss}")
        return avg_val_loss

    train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader()

    # Step 4: 训练和验证模型
    for epoch in range(config["max_epochs"]):
        print(f"Epoch {epoch + 1}/{config['max_epochs']}")
        train_loss = train_one_epoch(train_dataloader, model, guide, loss_module, optimizer, config)
        print("SUCCESSFUL!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        val_loss = validate(val_dataloader, model, guide, loss_module)


        # 可选：每个epoch后保存模型
        # mindspore.save_checkpoint(model, f"samsvae_model_epoch_{epoch+1}.ckpt")

    # Step 5: 保存最终模型
    # mindspore.save_checkpoint(model, "samsvae_model_final.ckpt")

    # Step 6: 加载模型（如有需要）
    # mindspore.load_checkpoint("samsvae_model_final.ckpt", net=model)

    # 可选：预测步骤可以根据config["predictor"]来实现

if __name__ == "__main__":
    train(config)