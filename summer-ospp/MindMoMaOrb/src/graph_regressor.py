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
"""GraphRegressor."""

from typing import Dict, Literal, Optional, Tuple, Union
import numpy

import mindspore as ms
from mindspore import Parameter, ops, Tensor, mint

from src import base, segment_ops
from src.gns import _KEY, MoleculeGNS
from src.nn_util import build_mlp
from src.property_definitions import PROPERTIES, PropertyDefinition
from src.reference_energies import get_reference_energies


class LinearReferenceEnergy(ms.nn.Cell):
    """Linear reference energy (no bias term)."""

    def __init__(
            self,
            weight_init: Optional[numpy.ndarray] = None,
            trainable: Optional[bool] = None,
    ):
        super().__init__()

        if trainable is None:
            trainable = weight_init is None

        self.linear = ms.nn.Dense(118, 1, has_bias=False)
        if weight_init is not None:
            self.linear.weight.set_data(Tensor(weight_init, dtype=ms.float32).reshape(1, 118))
        if not trainable:
            self.linear.weight.requires_grad = False

    def construct(self, atom_types: Tensor, n_node: Tensor) -> Tensor:
        """construct pass of the LinearReferenceEnergy.

        Args:
            atom_types: A tensor of atomic numbers of shape (n_atoms,)

        Returns:
            A tensor of shape (n_graphs, 1) containing the reference energy.
        """
        one_hot_atomic = ops.OneHot()(atom_types, 118, Tensor(1.0, ms.float32), Tensor(0.0, ms.float32))

        reduced = segment_ops.aggregate_nodes(one_hot_atomic, n_node, reduction="sum")
        #print(reduced.shape, self.linear.weight.shape)
        return self.linear(reduced)


class ScalarNormalizer(ms.nn.Cell):
    """Scalar normalizer that learns mean and std from data.

    NOTE: Multi-dimensional tensors are flattened before updating
    the running mean/std. This is desired behaviour for force targets.
    """

    def __init__(
            self,
            init_mean: Optional[Union[Tensor, float]] = None,
            init_std: Optional[Union[Tensor, float]] = None,
            init_num_batches: Optional[int] = 1000,
    ) -> None:
        """Initializes the ScalarNormalizer.

        To enhance training stability, consider setting an init mean + std.
        """
        super().__init__()
        self.bn = mint.nn.BatchNorm1d(1, affine=False, momentum=None)  # type: ignore
        self.bn.running_mean = Parameter(Tensor([0], ms.float32))
        self.bn.running_var = Parameter(Tensor([1], ms.float32))
        self.bn.num_batches_tracked = Parameter(Tensor([1000], ms.float32))
        self.stastics = {
            "running_mean": init_mean if init_mean is not None else 0.0,
            "running_var": init_std**2 if init_std is not None else 1.0,
            "num_batches_tracked": init_num_batches if init_num_batches is not None else 1000,
        }

    def construct(self, x: Tensor) -> Tensor:
        """Normalize by running mean and std."""
        if self.training:
            self.bn(x.view(-1, 1))
        if hasattr(self, "running_mean"):
            return (x - self.running_mean) / mint.sqrt(self.running_var)
        return (x - self.bn.running_mean) / mint.sqrt(self.bn.running_var)  # type: ignore

    def inverse(self, x: Tensor) -> Tensor:
        """Reverse the construct normalization."""
        if hasattr(self, "running_mean"):
            return x * mint.sqrt(self.running_var) + self.running_mean
        return x * mint.sqrt(self.bn.running_var) + self.bn.running_mean  # type: ignore


class NodeHead(ms.nn.Cell):
    """Node prediction head that can be appended to a base model.

    This head could be added to the foundation model to enable
    auxiliary tasks during pretraining, or added afterwards
    during a finetuning step.
    """

    def __init__(
            self,
            latent_dim: int,
            num_mlp_layers: int,
            mlp_hidden_dim: int,
            target: Union[str, PropertyDefinition],
            dropout: Optional[float] = None,
            remove_mean: bool = True,
    ):
        """Initializes the NodeHead MLP.

        Args:
            input_dim (int): Dimensionality of the incoming latent vector from the base model.
            num_mlp_layers (int): Number of MLP layers.
            mlp_hidden_dim (int): MLP hidden size.
            target: either the name of a PropertyDefinition or a PropertyDefinition itself.
            dropout: The level of dropout to apply.
            remove_mean: Whether to remove the mean of the node features.
        """
        super().__init__()
        if isinstance(target, str):
            target = PROPERTIES[target]
        self.target_property = target

        if target.domain != "real":
            raise ValueError("NodeHead only supports real targets.")

        if target.means is not None and target.stds is not None:
            self.normalizer = ScalarNormalizer(
                init_mean=target.means,
                init_std=target.stds,
            )
        else:
            self.normalizer = ScalarNormalizer()

        self.mlp = build_mlp(
            input_size=latent_dim,
            hidden_layer_sizes=[mlp_hidden_dim] * num_mlp_layers,
            output_size=self.target_property.dim,
            dropout=dropout,
        )

        self.remove_mean = remove_mean

    def construct(self, batch, nodes=None):
        """Predictions with raw logits (no sigmoid/softmax or any inverse transformations)."""
        feat = batch.node_features[_KEY] if nodes is None else nodes[_KEY]
        pred = self.mlp(feat)
        if self.remove_mean:
            system_means = segment_ops.aggregate_nodes(
                pred, batch.n_node, reduction="mean"
            )
            node_broadcasted_means = mint.repeat_interleave(
                system_means, batch.n_node, dim=0
            )
            pred = pred - node_broadcasted_means
        # batch.node_features["node_pred"] = pred
        res = {"node_pred": pred}
        return res

    def predict(self, batch, nodes=None) -> Tensor:
        """Predict node/edge/graph attribute."""
        out = self(batch, nodes)
        pred = out["node_pred"]
        return self.normalizer.inverse(pred)

    def loss(self, batch, out_batch=None):
        """Apply mlp to compute loss and metrics."""
        batch_n_node = batch.n_node
        assert batch.node_targets is not None
        target = batch.node_targets[self.target_property.name].squeeze(-1)
        # pred = batch.node_features["node_pred"].squeeze(-1)
        pred = out_batch["node_pred"].squeeze(-1)
        # make sure we remove fixed atoms before normalization
        pred, target, batch_n_node = _remove_fixed_atoms(
            pred, target, batch_n_node, batch.fix_atoms, self.training
        )
        mae = mint.abs(pred - self.normalizer(target))
        raw_pred = self.normalizer.inverse(pred)
        raw_mae = mint.abs(raw_pred - target)

        if self.target_property.dim > 1:
            mae = mae.mean(dim=-1)
            mae = segment_ops.aggregate_nodes(
                mae, batch_n_node, reduction="mean"
            ).mean()
            raw_mae = raw_mae.mean(dim=-1)
            raw_mae = segment_ops.aggregate_nodes(
                raw_mae, batch_n_node, reduction="mean"
            ).mean()
        else:
            mae = mae.mean()
            raw_mae = raw_mae.mean()
        metrics = {
            "node_mae": mae.item(),
            "node_mae_raw": raw_mae.item(),
            "node_cosine_sim": ops.cosine_similarity(raw_pred, target, dim=-1).mean().item(),
            "fwt_0.03": forces_within_threshold(raw_pred, target, batch_n_node),
        }
        return mae, base.ModelOutput(loss=mae, log=metrics)


class GraphHead(ms.nn.Cell):
    """MLP Regression head that can be appended to a base model.

    This head could be added to the foundation model to enable
    auxiliary tasks during pretraining, or added afterwards
    during a finetuning step.
    """

    def __init__(
            self,
            latent_dim: int,
            num_mlp_layers: int,
            mlp_hidden_dim: int,
            target: Union[str, PropertyDefinition],
            node_aggregation: Literal["sum", "mean"] = "mean",
            dropout: Optional[float] = None,
            compute_stress: Optional[bool] = False,
    ):
        """Initializes the GraphHead MLP.

        Args:
            input_dim (int): Dimensionality of the incoming latent vector from the base model.
            num_mlp_layers (int): Number of MLP layers.
            mlp_hidden_dim (int): MLP hidden size.
            target: either the name of a PropertyDefinition or a PropertyDefinition itself
            node_aggregation: The method for aggregating the node features
                from the pretrained model representations.
            dropout: The level of dropout to apply.
        """
        super().__init__()
        if isinstance(target, str):
            target = PROPERTIES[target]
        self.target_property = target

        if target.means is not None and target.stds is not None:
            self.normalizer = ScalarNormalizer(
                init_mean=target.means,
                init_std=target.stds,
            )
        else:
            self.normalizer = ScalarNormalizer()

        self.node_aggregation = node_aggregation
        self.mlp = build_mlp(
            input_size=latent_dim,
            hidden_layer_sizes=[mlp_hidden_dim] * num_mlp_layers,
            output_size=self.target_property.dim,
            dropout=dropout,
        )
        activation_dict = {
            "real": ops.Identity,
            "binary": ops.Sigmoid,
            "categorical": ops.Softmax,
        }
        self.output_activation = activation_dict[self.target_property.domain]()
        self.compute_stress = compute_stress

    def construct(self, batch, node_features=None):
        """Predictions with raw logits (no sigmoid/softmax or any inverse transformations)."""
        feat = batch.node_features[_KEY] if node_features is None else node_features[_KEY]

        # aggregate to get a tensor of shape (num_graphs, latent_dim)
        mlp_input = segment_ops.aggregate_nodes(
            feat,
            batch.n_node,
            reduction=self.node_aggregation,
        )

        pred = self.mlp(mlp_input)
        if self.compute_stress:
            # we need to name the stress prediction differently
            # batch.system_features["stress_pred"] = pred
            res = {"stress_pred": pred}
        else:
            # batch.system_features["graph_pred"] = pred
            res = {"graph_pred": pred}
        return res

    def predict(self, batch, node_features=None) -> Tensor:
        """Predict node/edge/graph attribute."""
        pred = self(batch, node_features)
        if self.compute_stress:
            logits = pred["stress_pred"].squeeze(-1)
        else:
            logits = pred["graph_pred"].squeeze(-1)
        probs = self.output_activation(logits)
        if self.target_property.domain == "real":
            probs = self.normalizer.inverse(probs)
        return probs

    def loss(self, batch, out_batch=None):
        """Apply mlp to compute loss and metrics.

        Depending on whether the target is real/binary/categorical, we
        use an MSE/cross-entropy loss. In the case of cross-entropy, the
        preds are logits (not normalised) to take advantage of numerically
        stable log-softmax.
        """
        assert batch.system_targets is not None
        target = batch.system_targets[self.target_property.name].squeeze(-1)
        if self.compute_stress:
            # pred = batch.system_features["stress_pred"].squeeze(-1)
            pred = out_batch["stress_pred"].squeeze(-1)
        else:
            # pred = batch.system_features["graph_pred"].squeeze(-1)
            pred = out_batch["graph_pred"].squeeze(-1)

        domain = self.target_property.domain
        # Short circuit for binary and categorical targets
        if domain == "binary":
            loss, metrics = bce_loss(pred, target, "graph")
            return base.ModelOutput(loss=loss, log=metrics)
        if domain == "categorical":
            loss, metrics = cross_entropy_loss(pred, target, "graph")
            return base.ModelOutput(loss=loss, log=metrics)

        normalized_target = self.normalizer(target)
        errors = normalized_target - pred
        mae = mint.abs(errors).mean()

        raw_pred = self.normalizer.inverse(pred)
        raw_mae = mint.abs(raw_pred - target).mean()
        if self.compute_stress:
            metrics = {"stress_mae": mae.item(), "stress_mae_raw": raw_mae.item()}
        else:
            metrics = {"graph_mae": mae.item(), "graph_mae_raw": raw_mae.item()}
        return mae, base.ModelOutput(loss=mae, log=metrics)


class EnergyHead(GraphHead):
    """Energy prediction head that can be appended to a base model."""

    def __init__(
            self,
            latent_dim: int,
            num_mlp_layers: int,
            mlp_hidden_dim: int,
            target: Union[str, PropertyDefinition] = "energy",
            predict_atom_avg: bool = True,
            reference_energy_name: str = "mp-traj-d3",  # or 'vasp-shifted'
            train_reference: bool = False,
            dropout: Optional[float] = None,
            node_aggregation: Optional[str] = None,
    ):
        """Initializes the EnergyHead MLP.

        Args:
            input_dim (int): Dimensionality of the incoming latent vector from the base model.
            num_mlp_layers (int): Number of MLP layers.
            mlp_hidden_dim (int): MLP hidden size.
            target: either the name of a PropertyDefinition or a PropertyDefinition itself.
            predict_atom_avg: Whether to predict the average atom energy or total.
            reference_energy_name: The name of the linear reference energy model to use.
            train_reference: Whether the reference energy params are learnable.
            dropout: The level of dropout to apply.
            node_aggregation: (deprecated) The method for aggregating the node features
        """
        ref = get_reference_energies(reference_energy_name)
        target = PROPERTIES[target] if isinstance(target, str) else target
        if predict_atom_avg:
            assert not node_aggregation or node_aggregation == "mean"
            target.means = Tensor([ref.residual_mean_per_atom])
            target.stds = Tensor([ref.residual_std_per_atom])
        else:
            assert not node_aggregation or node_aggregation == "sum"
            target.means = Tensor([ref.residual_mean])
            target.stds = Tensor([ref.residual_std])

        super().__init__(
            latent_dim=latent_dim,
            num_mlp_layers=num_mlp_layers,
            mlp_hidden_dim=mlp_hidden_dim,
            target=target,
            node_aggregation=node_aggregation,  # type: ignore
            dropout=dropout,
        )
        self.reference = LinearReferenceEnergy(
            weight_init=ref.coefficients, trainable=train_reference
        )
        self.atom_avg = predict_atom_avg

    def predict(self, batch, node_features=None) -> Tensor:
        """Predict energy."""
        pred = self(batch, node_features)["graph_pred"]
        pred = self.normalizer.inverse(pred).squeeze(-1)
        if self.atom_avg:
            pred = pred * batch.n_node
        pred = pred + self.reference(batch.atomic_numbers, batch.n_node)
        return pred

    def loss(self, batch, out_batch=None):
        """Apply mlp to compute loss and metrics."""
        assert batch.system_targets is not None
        target = batch.system_targets[self.target_property.name].squeeze(-1)
        # pred = batch.system_features["graph_pred"].squeeze(-1)
        pred = out_batch["graph_pred"].squeeze(-1)

        reference = self.reference(batch.atomic_numbers, batch.n_node).squeeze(-1)
        reference_target = target - reference
        if self.atom_avg:
            reference_target = reference_target / batch.n_node

        normalized_reference = self.normalizer(reference_target)
        model_loss = normalized_reference - pred

        raw_pred = self.normalizer.inverse(pred)
        if self.atom_avg:
            raw_pred = raw_pred * batch.n_node
        raw_mae = mint.abs((raw_pred + reference) - target).mean()

        reference_mae = mint.abs(reference_target).mean()
        model_mae = mint.abs(model_loss).mean()
        metrics = {
            "energy_reference_mae": reference_mae.item(),
            "energy_mae": model_mae.item(),
            "energy_mae_raw": raw_mae.item(),
        }
        return model_mae, base.ModelOutput(loss=model_mae, log=metrics)


class GraphRegressor(ms.nn.Cell):
    """Graph Regressor for finetuning.

    The GraphRegressor combines a pretrained base model
    with two regression heads for finetuning; one head for
    a node level task, and one for a graph level task.
    The base model can be optionally fine-tuned.
    The regression head is a linear/MLP transformation
    of the sum/avg of the graph's node activations.
    """

    def __init__(
            self,
            model: MoleculeGNS,
            node_head: Optional[NodeHead] = None,
            graph_head: Optional[GraphHead] = None,
            stress_head: Optional[GraphHead] = None,
            model_requires_grad: bool = True,
            cutoff_layers: Optional[int] = None,
    ) -> None:
        """Initializes the GraphRegressor.

        Args:
            node_head : The regression head to use for node prediction.
            graph_head: The regression head to use for graph prediction.
            model: An optional pre-constructed, pretrained model to use for transfer learning/finetuning.
            model_requires_grad: Whether the underlying model should
                be finetuned or not.
        """
        super().__init__()

        if (node_head is None) and (graph_head is None):
            raise ValueError("Must provide at least one node/graph head.")
        self.node_head = node_head
        self.graph_head = graph_head
        self.stress_head = stress_head
        self.cutoff_layers = cutoff_layers

        self.model = model

        if self.cutoff_layers is not None:
            if self.cutoff_layers > self.model.num_message_passing_steps:
                raise ValueError(
                    f"cutoff_layers ({self.cutoff_layers}) must be less than or equal to"
                    f" the number of message passing steps ({self.model.num_message_passing_steps})"
                )
            self.model.gnn_stacks = self.model.gnn_stacks[: self.cutoff_layers]
            self.model.num_message_passing_steps = self.cutoff_layers

        self.model_requires_grad = model_requires_grad

        if not model_requires_grad:
            for param in self.model.parameters():
                param.requires_grad = False

        self.loss_fn = ms.nn.MSELoss()

    def predict(self, batch) -> Dict[str, Tensor]:
        """Predict node and/or graph level attributes.

        Args:
            batch: A batch of graphs to run prediction on.

        Returns:
            A dictionary containing a node_pred tensor attribute with
            for node predictions, and a tensor of graph level predictions.
        """
        _, nodes = self.model(batch)

        output = {}
        output["graph_pred"] = self.graph_head.predict(batch, nodes)
        output["stress_pred"] = self.stress_head.predict(batch, nodes)
        output["node_pred"] = self.node_head.predict(batch, nodes)

        return output

    def construct(self, batch):
        """construct pass of GraphRegressor."""
        edges, nodes = self.model(batch)

        res = {"edges": edges, "nodes": nodes}
        res.update(self.graph_head(batch, nodes))
        res.update(self.stress_head(batch, nodes))
        res.update(self.node_head(batch, nodes))

        return res

    def loss(self, batch, label=None):
        """Loss function of GraphRegressor."""
        assert label is None, "GraphRegressor does not support labels."

        out = self(batch)
        loss = Tensor(0.0, ms.float32)
        metrics: Dict = {}

        loss1, graph_out = self.graph_head.loss(batch, out)
        metrics.update(graph_out.log)
        loss = loss.type_as(loss1) + loss1

        loss2, stress_out = self.stress_head.loss(batch, out)
        metrics.update(stress_out.log)
        loss = loss.type_as(loss2) + loss2

        loss3, node_out = self.node_head.loss(batch, out)
        metrics.update(node_out.log)
        loss = loss.type_as(loss3) + loss3

        metrics["loss"] = loss.item()
        return loss, metrics


def binary_accuracy(
        pred: Tensor, target: Tensor, threshold: float = 0.5
) -> float:
    """Calculate binary accuracy between 2 tensors.

    Args:
        pred: the prediction tensor.
        target: the tensor of target values.
        threshold: Binary classification threshold. Default 0.5.

    Returns:
        mean accuracy.
    """
    return ((pred > threshold) == target).to(ms.float32).mean().item()


def categorical_accuracy(pred: Tensor, target: Tensor) -> float:
    """Calculate accuracy for K class classification.

    Args:
        pred: the tensor of logits for K classes of shape (..., K)
        target: tensor of integer target values of shape (...)

    Returns:
        mean accuracy.
    """
    pred_labels = mint.argmax(pred, dim=-1)
    return (pred_labels == target).to(ms.float32).mean().item()


def error_within_threshold(
        pred: Tensor, target: Tensor, threshold: float = 0.02
) -> float:
    """Calculate MAE between 2 tensors within a threshold.

    Args:
        pred: the prediction tensor.
        target: the tensor of target values.
        threshold: margin threshold. Default 0.02 (derived from OCP metrics).

    Returns:
        Mean predictions within threshold.
    """
    error = mint.abs(pred - target)
    within_threshold = error < threshold
    return within_threshold.to(ms.float32).mean().item()


def forces_within_threshold(
        pred: Tensor,
        target: Tensor,
        batch_num_nodes: Tensor,
        threshold: float = 0.03,
) -> float:
    """Calculate MAE between batched graph tensors within a threshold.

    The predictions for a graph are counted as being within the threshold
    only if all nodes in the graph have predictions within the threshold.

    Args:
        pred: the prediction tensor.
        target: the tensor of target values.
        batch_num_nodes: A tensor containing the number of nodes per
            graph.
        threshold: margin threshold. Default 0.03 (derived from OCP metrics).

    Returns:
        Mean predictions within threshold.
    """
    # Shape (batch_num_nodes, 3)
    error = mint.abs(pred - target)
    # Shape (batch_num_nodes)
    largest_dim_fwt = error.max(-1)[0] < threshold

    count_within_threshold = segment_ops.aggregate_nodes(
        largest_dim_fwt.float(), batch_num_nodes, reduction="sum"
    )
    # count equals batch_num_nodes if all nodes within threshold
    return (count_within_threshold == batch_num_nodes).to(ms.float32).mean().item()


def energy_and_forces_within_threshold(
        pred_energy: Tensor,
        pred_forces: Tensor,
        target_energy: Tensor,
        target_forces: Tensor,
        batch_num_nodes: Tensor,
        fixed_atoms: Optional[Tensor] = None,
        threshold: Tuple[float, float] = (0.02, 0.03),
) -> float:
    """Calculate MAE between batched graph energies and forces within a threshold.

    The predictions for a graph are counted as being within the threshold
    only if all nodes in the graph have predictions within the threshold AND
    the energies are also within a threshold. A combo of the two above functions.

    Args:
        pred_*: the prediction tensors.
        target_*: the tensor of target values.
        batch_num_nodes: A tensor containing the number of nodes per
            graph.
        fixed_atoms: A tensor of bools indicating which atoms are fixed.
        threshold: margin threshold. Default (0.02, 0.03) (derived from OCP metrics).
    Returns:
        Mean predictions within threshold.
    """
    energy_err = mint.abs(pred_energy - target_energy)
    ewt = energy_err < threshold[0]

    forces_err = mint.abs(pred_forces - target_forces)
    largest_dim_fwt = forces_err.max(-1).values < threshold[1]

    working_largest_dim_fwt = largest_dim_fwt

    if fixed_atoms is not None:
        fixed_per_graph = segment_ops.aggregate_nodes(
            fixed_atoms.int(), batch_num_nodes, reduction="sum"
        )
        # remove the fixed atoms from the counts
        batch_num_nodes = batch_num_nodes - fixed_per_graph
        # remove the fixed atoms from the forces
        working_largest_dim_fwt = largest_dim_fwt[not fixed_atoms]

    force_count_within_threshold = segment_ops.aggregate_nodes(
        working_largest_dim_fwt.int(), batch_num_nodes, reduction="sum"
    )
    fwt = force_count_within_threshold == batch_num_nodes

    # count equals batch_num_nodes if all nodes within threshold
    return (fwt & ewt).to(ms.float32).mean().item()


def _remove_fixed_atoms(
        pred_node: Tensor,
        node_target: Tensor,
        batch_n_node: Tensor,
        fix_atoms: Optional[Tensor],
        training: bool,
):
    """We use inf targets on purpose to designate nodes for removal."""
    assert len(pred_node) == len(node_target)
    if fix_atoms is not None and not training:
        pred_node = pred_node[~fix_atoms]
        node_target = node_target[~fix_atoms]
        batch_n_node = segment_ops.aggregate_nodes(
            (~fix_atoms).int(), batch_n_node, reduction="sum"
        )
    return pred_node, node_target, batch_n_node


def bce_loss(
        pred: Tensor, target: Tensor, metric_prefix: str = ""
) -> Tuple:
    """Binary cross-entropy loss with accuracy metric."""
    loss = mint.nn.BCEWithLogitsLoss()(pred, target.float())
    accuracy = binary_accuracy(pred, target)
    return (
        loss,
        {
            f"{metric_prefix}_accuracy": accuracy,
            f"{metric_prefix}_loss": loss.item(),
        },
    )


def cross_entropy_loss(
        pred: Tensor, target: Tensor, metric_prefix: str = ""
) -> Tuple:
    """Cross-entropy loss with accuracy metric."""
    loss = mint.nn.CrossEntropyLoss()(pred, target.long())
    accuracy = categorical_accuracy(pred, target)
    return (
        loss,
        {
            f"{metric_prefix}_accuracy": accuracy,
            f"{metric_prefix}_loss": loss.item(),
        },
    )
