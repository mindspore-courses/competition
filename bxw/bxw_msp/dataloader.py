from typing import Any, Dict, List, Literal, Iterable, Optional, Sequence, Tuple, Union, Mapping, TypedDict
import numpy as np
import mindspore
import mindspore.dataset as ds
import anndata
import pandas as pd
from distribution import *
from sklearn.model_selection import train_test_split

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