import mindspore
from typing import Any, Dict, List, Literal, Iterable, Optional, Sequence, Tuple, Union, Mapping, TypedDict
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
            "n": 32,
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