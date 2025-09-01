# ============================================================================
# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""pretrained."""

import os
from typing import Optional
from functools import partial

from mindspore import nn, load_checkpoint, load_param_into_net

from src.featurization_utilities import (
    gaussian_basis_function,
)
from src.gns import MoleculeGNS
from src.graph_regressor import (
    EnergyHead,
    GraphHead,
    GraphRegressor,
    NodeHead,
)
from src.rbf import ExpNormalSmearing


def get_base(
        latent_dim: int = 256,
        mlp_hidden_dim: int = 512,
        num_message_passing_steps: int = 15,
        num_edge_in_features: int = 23,
        distance_cutoff: bool = True,
        attention_gate: str = "sigmoid",
        rbf_transform: str = "gaussian",
) -> MoleculeGNS:
    """Define the base pretrained model architecture."""
    return MoleculeGNS(
        num_node_in_features=256,
        num_node_out_features=3,
        num_edge_in_features=num_edge_in_features,
        latent_dim=latent_dim,
        interactions="simple_attention",
        interaction_params={
            "distance_cutoff": distance_cutoff,
            "polynomial_order": 4,
            "cutoff_rmax": 6,
            "attention_gate": attention_gate,
        },
        num_message_passing_steps=num_message_passing_steps,
        num_mlp_layers=2,
        mlp_hidden_dim=mlp_hidden_dim,
        rbf_transform=(
            ExpNormalSmearing(num_rbf=50, cutoff_upper=10.0)
            if rbf_transform == "exp_normal_smearing"
            else partial(gaussian_basis_function, num_bases=20, radius=10.0)
        ),
        use_embedding=True,
        node_feature_names=["feat"],
        edge_feature_names=["feat"],
    )


def load_model_for_inference(model: nn.Cell, weights_path: str) -> nn.Cell:
    """
    Load a pretrained model in inference mode, using GPU if available.
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Checkpoint file {weights_path} not found.")
    param_dict = load_checkpoint(weights_path)

    try:
        load_param_into_net(model, param_dict)
    except ValueError:
        print("Warning: The checkpoint file has more parameters than the model. \
              This may be due to a mismatch in the model architecture or version.")
        params = []
        for key in param_dict:
            params.append(param_dict[key])
        for parameters in model.trainable_params():
            param_ckpt = params.pop(0)
            assert parameters.shape == param_ckpt.shape, f"Shape mismatch: {parameters.name}"
            param_ckpt = param_ckpt.reshape(parameters.shape)
            parameters.set_data(param_ckpt)

    model.set_train(False)
    return model

def orb_v2(
        weights_path: Optional[str] = None,
):
    """Load ORB v2."""
    base = get_base()

    model = GraphRegressor(
        graph_head=EnergyHead(
            latent_dim=256,
            num_mlp_layers=1,
            mlp_hidden_dim=256,
            target="energy",
            node_aggregation="mean",
            reference_energy_name="vasp-shifted",
            train_reference=True,
            predict_atom_avg=True,
        ),
        node_head=NodeHead(
            latent_dim=256,
            num_mlp_layers=1,
            mlp_hidden_dim=256,
            target="forces",
            remove_mean=True,
        ),
        stress_head=GraphHead(
            latent_dim=256,
            num_mlp_layers=1,
            mlp_hidden_dim=256,
            target="stress",
            compute_stress=True,
        ),
        model=base,
    )
    model = load_model_for_inference(model, weights_path)
    return model


def orb_mptraj_only_v2(
        weights_path: Optional[str] = None,
):
    """Load ORB MPTraj Only v2."""

    return orb_v2(weights_path,)


def orb_d3_v2(
        weights_path: Optional[str] = None,
):
    """Load ORB D3 v2."""
    base = get_base()

    model = GraphRegressor(
        graph_head=EnergyHead(
            latent_dim=256,
            num_mlp_layers=1,
            mlp_hidden_dim=256,
            target="energy",
            node_aggregation="mean",
            reference_energy_name="vasp-shifted",
            train_reference=True,
            predict_atom_avg=True,
        ),
        node_head=NodeHead(
            latent_dim=256,
            num_mlp_layers=1,
            mlp_hidden_dim=256,
            target="forces",
            remove_mean=True,
        ),
        stress_head=GraphHead(
            latent_dim=256,
            num_mlp_layers=1,
            mlp_hidden_dim=256,
            target="stress",
            compute_stress=True,
        ),
        model=base,
    )

    model = load_model_for_inference(model, weights_path)

    return model


def orb_d3_sm_v2(
        weights_path: Optional[str] = None,
):
    """Load ORB D3 v2."""
    base = get_base(
        num_message_passing_steps=10,
    )

    model = GraphRegressor(
        graph_head=EnergyHead(
            latent_dim=256,
            num_mlp_layers=1,
            mlp_hidden_dim=256,
            target="energy",
            node_aggregation="mean",
            reference_energy_name="vasp-shifted",
            train_reference=True,
            predict_atom_avg=True,
        ),
        node_head=NodeHead(
            latent_dim=256,
            num_mlp_layers=1,
            mlp_hidden_dim=256,
            target="forces",
            remove_mean=True,
        ),
        stress_head=GraphHead(
            latent_dim=256,
            num_mlp_layers=1,
            mlp_hidden_dim=256,
            target="stress",
            compute_stress=True,
        ),
        model=base,
    )

    model = load_model_for_inference(model, weights_path)

    return model


def orb_d3_xs_v2(
        weights_path: Optional[str] = None,
):
    """Load ORB D3 xs v2."""
    base = get_base(
        num_message_passing_steps=5,
    )

    model = GraphRegressor(
        graph_head=EnergyHead(
            latent_dim=256,
            num_mlp_layers=1,
            mlp_hidden_dim=256,
            target="energy",
            node_aggregation="mean",
            reference_energy_name="vasp-shifted",
            train_reference=True,
            predict_atom_avg=True,
        ),
        node_head=NodeHead(
            latent_dim=256,
            num_mlp_layers=1,
            mlp_hidden_dim=256,
            target="forces",
            remove_mean=True,
        ),
        stress_head=GraphHead(
            latent_dim=256,
            num_mlp_layers=1,
            mlp_hidden_dim=256,
            target="stress",
            compute_stress=True,
        ),
        model=base,
    )

    model = load_model_for_inference(model, weights_path)

    return model


ORB_PRETRAINED_MODELS = {
    "orb-v2": orb_v2,
    "orb-d3-v2": orb_d3_v2,
    "orb-d3-sm-v2": orb_d3_sm_v2,
    "orb-d3-xs-v2": orb_d3_xs_v2,
    "orb-mptraj-only-v2": orb_mptraj_only_v2,
}
