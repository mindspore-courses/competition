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
"""Test module for ORB model components and operations.

This module contains test cases for various components of the ORB model including:
- Graph building
- Attention mechanism
- GNS (Graph Neural Network)
- Model inference
- Loss computation
- Segment operations
"""

import sys
import pickle
import numpy as np
from functools import partial
import importlib

import ase
import ase.db
import mindspore as ms
from mindspore import Tensor, ops, context, load_checkpoint, load_param_into_net
from mindspore import dtype as mstype

sys.path.append("../mindspore/orb_models/")
from src import atomic_system, base, pretrained, segment_ops
from src.gns import MoleculeGNS, AttentionInteractionNetwork
from src.rbf import ExpNormalSmearing
from src.featurization_utilities import gaussian_basis_function
from src.ase_dataset import AseSqliteDataset, BufferData
from src.segment_ops import segment_sum, segment_softmax
from utils import *

class RenameUnpickler(pickle.Unpickler):
    """Custom unpickler to handle module renaming."""
    def find_class(self, module, name):
        print(f"Looking for: module={module}, name={name}")
        renamed_module = module
        if module == "orb_models.forcefield.base":
            renamed_module = "src.base"
            
        try:
            module_obj = importlib.import_module(renamed_module)
            return getattr(module_obj, name)
        except Exception as e:
            print(f"Error importing {renamed_module}: {e}")
            try:
                return super().find_class(module, name)
            except Exception as e2:
                print(f"Error with original module {module}: {e2}")
                raise
    

def load_graph_data(pkl_path: str):
    """Load graph data from pickle file.
    
    Args:
        pkl_path: Path to the pickle file
        
    Returns:
        tuple: (atoms, input_graph_ms, output_graph_np)
    """
    with open(pkl_path, "rb") as f:
        loaded = pickle.load(f)
        # loaded = RenameUnpickler(f).load()

    atoms = loaded["atoms"]
    input_graph_np = loaded["input_graph"]
    output_graph_np = loaded["output_graph"]

    input_graph_ms = base.AtomGraphs(
        *[numpy_to_tensor(getattr(input_graph_np, field))
        for field in input_graph_np._fields]
    )

    return atoms, input_graph_ms, output_graph_np


def test_build_graph():
    """Test ase_atoms_to_atom_graphs function."""
    print("\n=== Testing build_graph ===")
    atoms, input_graph_ms, _ = load_graph_data("gns_input_output_ms.pkl")

    input_graph_np = base.AtomGraphs(*[tensor_to_numpy(getattr(input_graph_ms, field)) for field in input_graph_ms._fields])

    atom_graph_ms = atomic_system.ase_atoms_to_atom_graphs(
        atoms,
        brute_force_knn=False,
    )
    atom_graph_ms = base.AtomGraphs(*[tensor_to_numpy(getattr(atom_graph_ms, field)) for field in atom_graph_ms._fields])
    
    result = is_equal(atom_graph_ms, input_graph_np)
    print(f"Test build_graph {'passed' if result else 'failed'}")


def test_attn():
    """Test attention network."""
    print("\n=== Testing attention network ===")
    # prepare data
    _, input_graph_ms, output_graph_np = load_graph_data("attn_input_output_ms.pkl")

    attn_net = AttentionInteractionNetwork(
        num_node_in=256,
        num_node_out=256,
        num_edge_in=256,
        num_edge_out=256,
        num_mlp_layers=2,
        mlp_hidden_dim=512,
    )

    # load checkpoint
    parms = []
    param_dict = load_checkpoint("orb_ms.ckpt")
    param_dict = {k: v for k, v in param_dict.items() if "gnn_stacks.0." in k}
    for key in param_dict:
        parms.append(param_dict[key])
    for parameters in attn_net.trainable_params():
        parameters.set_data(parms.pop(0))

    # inference
    edges, nodes = attn_net(input_graph_ms, input_graph_ms.edge_features, input_graph_ms.node_features)
    out = input_graph_ms.clone()
    out = out._replace(
        node_features={**nodes},
        edge_features={**edges},
    )
    out = base.AtomGraphs(
        *[tensor_to_numpy(getattr(out, field))
        for field in out._fields]
    )

    # Validate results
    out_node_feats = out.node_features["feat"]
    out_edge_feats = out.edge_features["feat"]
    out_node_feats_np = output_graph_np.node_features["feat"]
    out_edge_feats_np = output_graph_np.edge_features["feat"]
    
    print(f"Node feature MAE: {np.mean(np.abs(out_node_feats - out_node_feats_np)):.6f}")
    print(f"Edge feature MAE: {np.mean(np.abs(out_edge_feats - out_edge_feats_np)):.6f}")

    result = is_equal(out, output_graph_np)
    assert result, "Attention network output mismatch"
    print("Test attention network passed")


def test_gns():
    """Test MoleculeGNS network."""
    print("\n=== Testing MoleculeGNS network ===")
    _, input_graph_ms, output_graph_pt = load_graph_data("gns_input_output_ms.pkl")

    # load gns model and checkpoint
    rbf_transform = "gaussian" # "exp_normal_smearing"
    gns_model = MoleculeGNS(
        num_node_in_features=256,
        num_node_out_features=3,
        num_edge_in_features=23,
        latent_dim=256,
        interactions="simple_attention",
        interaction_params={
            "distance_cutoff": True,
            "polynomial_order": 4,
            "cutoff_rmax": 6,
            "attention_gate": "sigmoid",
        },
        num_message_passing_steps=15,
        num_mlp_layers=2,
        mlp_hidden_dim=512,
        rbf_transform=(
            ExpNormalSmearing(num_rbf=50, cutoff_upper=10.0)
            if rbf_transform == "exp_normal_smearing"
            else partial(gaussian_basis_function, num_bases=20, radius=10.0)
        ),
        use_embedding=True,
        node_feature_names=["feat"],
        edge_feature_names=["feat"],
    )
    # load checkpoint
    path = "orb_ms.ckpt"
    parms = []
    param_dict = load_checkpoint(path)
    for key in param_dict:
        parms.append(param_dict[key])
    for parameters in gns_model.trainable_params():
        parameters.set_data(parms.pop(0))

    edges, nodes = gns_model(input_graph_ms)
    out = input_graph_ms.clone()
    out = out._replace(
        node_features={**nodes},
        edge_features={**edges},
    )
    out = base.AtomGraphs(
        *[tensor_to_numpy(getattr(out, field))
        for field in out._fields]
    )

    # alignment feature
    node_out_ms = out.node_features["feat"]
    edge_out_ms = out.edge_features["feat"]
    node_out_pt = output_graph_pt.node_features["feat"]
    edge_out_pt = output_graph_pt.edge_features["feat"]
    print(np.max(np.abs(node_out_ms - node_out_pt)))
    print(np.max(np.abs(edge_out_ms - edge_out_pt)))

    flag = is_equal(out, output_graph_pt)
    assert flag, "MoleculeGNS network output mismatch"
    print("Test MoleculeGNS network passed")


def test_inference():
    """Test Orb network inference."""
    print("\n=== Testing Orb network inference ===")
    # load data
    # with open("inference_input_output.pkl", "rb") as f:
    #     loaded = RenameUnpickler(f).load()

    with open("orb_input_output.pkl", "rb") as f:
        loaded = pickle.load(f)
    
    # for k in loaded:
    #     print(f"Loaded key: {k}, type: {type(loaded[k])}")

    atoms = loaded["atoms"]
    input_graph = loaded["input_graph"]
    output_pt = loaded["output"]

    # with open("orb_input_output.pkl", "wb") as f:
    #     pickle.dump(loaded, f)

    atom_graph_ms = atomic_system.ase_atoms_to_atom_graphs(
        atoms,
        brute_force_knn=False,
    )

    # load model
    orb_path = "/home/cjh/orb/mindspore/orb_models/orb_ckpts/orb-mptraj-only-v2-20250524.ckpt"
    regressor = pretrained.orb_mptraj_only_v2(weights_path=orb_path) # GNS + HEAD
    regressor.set_train(False)  # set to eval mode

    # inference
    out_ms = regressor.predict(atom_graph_ms) # a dict
    out_ms = {k: tensor_to_numpy(v) for k, v in out_ms.items()}
    print(out_ms)
    
    # accuracy alignment
    for k in out_ms:
        flag = compare_output(out_ms[k], output_pt[k])
        assert flag, f"Orb network inference output {k} mismatch"
    print("Test Orb network inference passed")


def test_loss():
    """Test Orb network loss computation."""
    print("\n=== Testing Orb network loss computation ===")
    # load data
    with open("loss_input_output_ms.pkl", "rb") as f:
        loaded = pickle.load(f)

    input_graph_np = loaded["input_graph"]
    loss_pt = loaded["loss"]

    input_graph_ms = base.AtomGraphs(
        *[numpy_to_tensor(getattr(input_graph_np, field))
        for field in input_graph_np._fields]
    )

    # load model
    orb_path = "/home/cjh/orb/mindspore/orb_models/orb_ckpts/orb-mptraj-only-v2-20250524.ckpt"
    regressor = pretrained.orb_mptraj_only_v2(weights_path=orb_path) # GNS + HEAD
   
    # inference
    # regressor.set_train()
    regressor.set_train(False)
    _, log = regressor.loss(input_graph_ms) # a dict
    flag = is_equal(
        np.array([log[k] for k in log]),
        np.array([loss_pt[k] for k in loss_pt])
    )
    assert flag, "Orb network loss computation output mismatch"
    print("Test Orb network loss computation passed")


def test_segment_sum():
    """Test segment sum operation."""
    print("\n=== Testing segment_sum ===")
    data = np.load('segment_sum_input_output.npz')
    input1, input2, input3 = data['first_input'], data['second_input'], data['third_input']
    y = data['sent_attributes']
    res = segment_sum(Tensor(input1), Tensor(input2), input3.item()).numpy()
    flag = compare_output(y, res, FP32_ATOL, FP32_RTOL)
    assert flag, "Segment sum test failed"
    print("Test segment_sum passed")


def test_segment_softmax():
    """Test segment softmax operation."""
    print("\n=== Testing segment_softmax ===")
    data = np.load('segment_softmax_input_output.npz')
    input1, input2, input3, input4 = data['first_input'], data['second_input'], data['third_input'], data['fourth_input']
    y = data['output']
    res = segment_softmax(Tensor(input1), Tensor(input2), input3.item(), Tensor(input4)).numpy()
    flag = compare_output(y, res, FP32_ATOL, FP32_RTOL)
    assert flag, "Segment softmax test failed"
    print("Test segment_softmax passed")


if __name__ == "__main__":
    context.set_context(pynative_synchronize=True, device_target='Ascend', device_id=1)
    
    # Run tests
    test_build_graph()
    test_attn()
    test_gns()
    test_inference()
    test_loss()
    test_segment_sum()
    test_segment_softmax()