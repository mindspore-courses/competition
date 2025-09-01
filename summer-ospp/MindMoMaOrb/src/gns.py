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
"""GNS Molecule."""


from typing import List, Literal, Optional, Dict, Any

import numpy as np
from mindspore import nn, ops, Tensor
from mindspore.common.initializer import Uniform
import mindspore.ops.operations as P

from src.nn_util import build_mlp

_KEY = "feat"


def mlp_and_layer_norm(in_dim: int, out_dim: int, hidden_dim: int, n_layers: int) -> nn.SequentialCell:
    """Create an MLP followed by layer norm."""

    layers = build_mlp(
        in_dim,
        [hidden_dim for _ in range(n_layers)],
        out_dim,
    )
    layers.append(nn.LayerNorm((out_dim,)))
    return layers


def get_cutoff(p: int, r: Tensor, r_max: float) -> Tensor:
    """Get the cutoff function for attention."""
    envelope = 1.0 - ((p + 1.0) * (p + 2.0) / 2.0) * ops.pow(r / r_max, p) + \
        p * (p + 2.0) * ops.pow(r / r_max, p + 1) - \
        (p * (p + 1.0) / 2) * ops.pow(r / r_max, p + 2)
    cutoff = ops.expand_dims(
        ops.where(r < r_max, envelope, ops.zeros_like(envelope)), -1)
    return cutoff


class AtomEmbedding(nn.Cell):
    """Initial atom embeddings based on the atom type."""

    def __init__(self, emb_size, num_elements):
        super().__init__()
        self.emb_size = emb_size
        self.embeddings = nn.Embedding(
            num_elements + 1, emb_size, embedding_table=Uniform(np.sqrt(3)))

    def construct(self, x):
        """Forward pass of the atom embedding layer."""
        h = self.embeddings(x)
        return h


class Encoder(nn.Cell):
    """Graph network encoder. Encode nodes and edges states to an MLP."""

    def __init__(self,
                 num_node_in_features: int,
                 num_node_out_features: int,
                 num_edge_in_features: int,
                 num_edge_out_features: int,
                 num_mlp_layers: int,
                 mlp_hidden_dim: int,
                 node_feature_names: List[str],
                 edge_feature_names: List[str]):
        super().__init__()
        self.node_feature_names = node_feature_names
        self.edge_feature_names = edge_feature_names
        self._node_fn = mlp_and_layer_norm(
            num_node_in_features, num_node_out_features, mlp_hidden_dim, num_mlp_layers)
        self._edge_fn = mlp_and_layer_norm(
            num_edge_in_features, num_edge_out_features, mlp_hidden_dim, num_mlp_layers)

    def construct(self, graph, node_features=None, edge_features=None):
        edges = graph.edge_features if edge_features is None else edge_features
        nodes = graph.node_features if node_features is None else node_features
        edge_features = ops.cat([edges[k] for k in self.edge_feature_names], axis=-1)
        node_features = ops.cat([nodes[k] for k in self.node_feature_names], axis=-1)

        edges.update({_KEY: self._edge_fn(edge_features)})
        nodes.update({_KEY: self._node_fn(node_features)})
        return edges, nodes


class InteractionNetwork(nn.Cell):
    """Interaction Network."""

    def __init__(self,
                 num_node_in: int,
                 num_node_out: int,
                 num_edge_in: int,
                 num_edge_out: int,
                 num_mlp_layers: int,
                 mlp_hidden_dim: int):
        super().__init__()
        self._node_mlp = mlp_and_layer_norm(
            num_node_in + num_edge_out, num_node_out, mlp_hidden_dim, num_mlp_layers)
        self._edge_mlp = mlp_and_layer_norm(
            num_node_in + num_node_in + num_edge_in, num_edge_out, mlp_hidden_dim, num_mlp_layers)

    def construct(self, graph, graph_edges=None, graph_nodes=None):
        """Forward pass of the interaction network."""
        nodes = graph.node_features[_KEY] if graph_nodes is None else graph_nodes[_KEY]
        edges = graph.edge_features[_KEY] if graph_edges is None else graph_edges[_KEY]
        senders = graph.senders
        receivers = graph.receivers

        sent_attributes = ops.gather(nodes, senders, 0)
        received_attributes = ops.gather(nodes, receivers, 0)

        edge_features = ops.cat(
            [edges, sent_attributes, received_attributes], axis=1)
        updated_edges = self._edge_mlp(edge_features)

        received_attributes = ops.scatter_add(
            ops.zeros_like(nodes), receivers, updated_edges)

        node_features = ops.cat([nodes, received_attributes], axis=1)
        updated_nodes = self._node_mlp(node_features)

        nodes = graph_nodes[_KEY] + updated_nodes
        edges = graph_edges[_KEY] + updated_edges

        node_features = {**graph.node_features, _KEY: nodes}
        edge_features = {**graph.edge_features, _KEY: edges}
        return edge_features, node_features

class AttentionInteractionNetwork(nn.Cell):
    """Attention Interaction Network."""

    def __init__(self,
                 num_node_in: int,
                 num_node_out: int,
                 num_edge_in: int,
                 num_edge_out: int,
                 num_mlp_layers: int,
                 mlp_hidden_dim: int,
                 attention_gate: Literal["sigmoid", "softmax"] = "sigmoid",
                 distance_cutoff: bool = True,
                 polynomial_order: Optional[int] = 4,
                 cutoff_rmax: Optional[float] = 6.0):
        super().__init__()
        self._num_node_in = num_node_in
        self._num_node_out = num_node_out
        self._num_edge_in = num_edge_in
        self._num_edge_out = num_edge_out
        self._num_mlp_layers = num_mlp_layers
        self._mlp_hidden_dim = mlp_hidden_dim
        self._node_mlp = mlp_and_layer_norm(
            num_node_in + num_edge_out + num_edge_out, num_node_out, mlp_hidden_dim, num_mlp_layers)
        self._edge_mlp = mlp_and_layer_norm(
            num_node_in + num_node_in + num_edge_in, num_edge_out, mlp_hidden_dim, num_mlp_layers)
        self._receive_attn = nn.Dense(num_edge_in, 1)
        self._send_attn = nn.Dense(num_edge_in, 1)
        self._distance_cutoff = distance_cutoff
        self._r_max = cutoff_rmax
        self._polynomial_order = polynomial_order
        self._attention_gate = attention_gate

        self.scatter_add = P.TensorScatterAdd()

    def construct(self, graph, graph_edges=None, graph_nodes=None):
        """Forward pass of the attention interaction network."""
        nodes = graph.node_features[_KEY] if graph_nodes is None else graph_nodes[_KEY]
        edges = graph.edge_features[_KEY] if graph_edges is None else graph_edges[_KEY]
        senders = graph.senders
        receivers = graph.receivers

        p = self._polynomial_order
        r_max = self._r_max
        r = graph.edge_features['r']
        cutoff = get_cutoff(p, r, r_max)

        sent_attributes = ops.gather(nodes, senders, 0)
        received_attributes = ops.gather(nodes, receivers, 0)

        if self._attention_gate == "softmax":
            receive_attn = ops.softmax(self._receive_attn(edges), axis=0)
            send_attn = ops.softmax(self._send_attn(edges), axis=0)
        else:
            receive_attn = ops.sigmoid(self._receive_attn(edges))
            send_attn = ops.sigmoid(self._send_attn(edges))

        if self._distance_cutoff:
            receive_attn = receive_attn * cutoff
            send_attn = send_attn * cutoff

        edge_features = ops.cat(
            [edges, sent_attributes, received_attributes], axis=1)
        updated_edges = self._edge_mlp(edge_features)

        if senders.ndim < 2:
            senders = senders.unsqueeze(-1)
        sent_attributes = self.scatter_add(
            ops.zeros_like(nodes), senders, updated_edges * send_attn)
        if receivers.ndim < 2:
            receivers = receivers.unsqueeze(-1)
        received_attributes = self.scatter_add(
            ops.zeros_like(nodes), receivers, updated_edges * receive_attn)

        node_features = ops.cat(
            [nodes, received_attributes, sent_attributes], axis=1)
        updated_nodes = self._node_mlp(node_features)

        nodes = graph_nodes[_KEY] + updated_nodes
        edges = graph_edges[_KEY] + updated_edges

        node_features = {**graph.node_features, _KEY: nodes}
        edge_features = {**graph.edge_features, _KEY: edges}
        return edge_features, node_features

class Decoder(nn.Cell):
    """The Decoder."""

    def __init__(self,
                 num_node_in: int,
                 num_node_out: int,
                 num_mlp_layers: int,
                 mlp_hidden_dim: int,
                 batch_norm: bool = False):
        super().__init__()
        seq = build_mlp(
            num_node_in,
            [mlp_hidden_dim for _ in range(num_mlp_layers)],
            num_node_out,
        )
        if batch_norm:
            seq.append(nn.BatchNorm1d(num_node_out))
        self.node_fn = nn.SequentialCell(seq)

    def construct(self, graph, graph_nodes=None):
        """Forward pass of the decoder."""
        nodes = graph.node_features[_KEY] if graph_nodes is None else graph_nodes[_KEY]
        updated = self.node_fn(nodes)
        return {**graph_nodes, "pred": updated}


class MoleculeGNS(nn.Cell):
    """GNS that works on molecular data."""

    def __init__(self,
                 num_node_in_features: int,
                 num_node_out_features: int,
                 num_edge_in_features: int,
                 latent_dim: int,
                 num_message_passing_steps: int,
                 num_mlp_layers: int,
                 mlp_hidden_dim: int,
                 node_feature_names: List[str],
                 edge_feature_names: List[str],
                 rbf_transform: nn.Cell,
                 use_embedding: bool = True,
                 interactions: Literal["default",
                                       "simple_attention"] = "simple_attention",
                 interaction_params: Optional[Dict[str, Any]] = None):
        super().__init__()
        self._encoder = Encoder(
            num_node_in_features=num_node_in_features,
            num_node_out_features=latent_dim,
            num_edge_in_features=num_edge_in_features,
            num_edge_out_features=latent_dim,
            num_mlp_layers=num_mlp_layers,
            mlp_hidden_dim=mlp_hidden_dim,
            node_feature_names=node_feature_names,
            edge_feature_names=edge_feature_names
        )
        if interactions == "default":
            InteractionNetworkClass = InteractionNetwork
        elif interactions == "simple_attention":
            InteractionNetworkClass = AttentionInteractionNetwork
        self.num_message_passing_steps = num_message_passing_steps
        if interaction_params is None:
            interaction_params = {}
        self.gnn_stacks = nn.CellList([
            InteractionNetworkClass(
                num_node_in=latent_dim,
                num_node_out=latent_dim,
                num_edge_in=latent_dim,
                num_edge_out=latent_dim,
                num_mlp_layers=num_mlp_layers,
                mlp_hidden_dim=mlp_hidden_dim,
                **interaction_params
            ) for _ in range(self.num_message_passing_steps)
        ])
        self._decoder = Decoder(
            num_node_in=latent_dim,
            num_node_out=num_node_out_features,
            num_mlp_layers=num_mlp_layers,
            mlp_hidden_dim=mlp_hidden_dim
        )
        self.rbf = rbf_transform
        self.use_embedding = use_embedding
        if self.use_embedding:
            self.atom_emb = AtomEmbedding(latent_dim, 118)

    def construct(self, batch):
        """Forward pass of the GNS."""
        edge_features = self.featurize_edges(batch)
        node_features = self.featurize_nodes(batch)
        edges, nodes = self._encoder(batch, node_features, edge_features)
        for gnn in self.gnn_stacks:
            edges, nodes = gnn(batch, edges, nodes)
        nodes = self._decoder(batch, nodes)
        return edges, nodes

    def featurize_nodes(self, batch):
        """Featurize the nodes of a graph."""
        one_hot_atomic = ops.OneHot()(
            batch.node_features["atomic_numbers"], 118, Tensor(1.0), Tensor(0.0)
        )
        if self.use_embedding:
            atomic_embedding = self.atom_emb(batch.node_features["atomic_numbers"])
        else:
            atomic_embedding = one_hot_atomic

        node_features = {**batch.node_features, **{_KEY: atomic_embedding}}
        return node_features

    def featurize_edges(self, batch):
        """Featurize the edges of a graph."""
        lengths = ops.norm(batch.edge_features['vectors'], dim=1)
        non_zero_divisor = ops.where(
            lengths == 0, ops.ones_like(lengths), lengths)
        unit_vectors = batch.edge_features['vectors'] / ops.expand_dims(non_zero_divisor, 1)
        rbfs = self.rbf(lengths)
        edge_features = ops.cat([rbfs, unit_vectors], axis=1)

        edge_features = {**batch.edge_features, **{_KEY: edge_features}}
        return edge_features
