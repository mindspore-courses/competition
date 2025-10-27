from __future__ import annotations  # noqa: F407

from copy import deepcopy
from functools import partial
from typing import Any, Callable, Dict, List, Tuple, Union

import mindspore as ms
import numpy as np

from mindspore import nn,ops

from models import layers
from models.layers import MSDeformAttn

class MLFusion(nn.Cell):
    def __init__(self,
                 d_model: int = 256,
                 d_ffn: int = 1024,
                 n_levels: int = 1,
                 n_heads: int = 1,
                 n_points: int = 1,
                 ffn_layer: str = 'Linear',
                 activation: str = 'ReLU',
                 dropout: float = 0.0,
                 norm: bool = False,
                 **kwargs):
        """Multi-Level Fusion Transformer.

        Arguments:
            d_model: Hidden feature (channel) dimension of the
                attention modules (self and cross attention).
            d_ffn: Hidden feature (channel) dimension of the
                feed forward module.
            n_levels: Number of feature levels of the cross attention module.
            n_heads: Number of attention heads of the attention modules
                (self and cross attention).
            n_points: Number of sampling points per attention head
                (self and cross attention) and per feature level (cross attention).
        """
        # Initialize base class
        super().__init__()

        # Initialize instance attributes
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.ffn_layer = ffn_layer
        self.activation = activation
        self.dropout = dropout
        self.norm = norm

        # Initialize self attention module
        self.self_attn = nn.MultiheadAttention(self.d_model, self.n_heads,
                                               dropout=self.dropout, batch_first=True)
        self.dropout1 = nn.Dropout(self.dropout,p=self.dropout)
        self.norm1 = nn.LayerNorm([self.d_model])

        # Initialize deformable cross attention module
        self.flatten1 = nn.Flatten(start_dim=1, end_dim=2)
        self.ms_deform_attn = MSDeformAttn(self.d_model, self.n_levels,
                                           self.n_heads, self.n_points)
        self.dropout2 = nn.Dropout(self.dropout,p=self.dropout)
        self.norm2 = nn.LayerNorm([self.d_model])

        # Initialize feed forward (ffn) module
        self.ffn1 = self._get_ffn_layer(self.ffn_layer, self.d_model, self.d_ffn)
        self.activation1 = self._get_activation_fn(self.activation)
        self.dropout3 = nn.Dropout(self.dropout,p=self.dropout)
        self.ffn2 = self._get_ffn_layer(self.ffn_layer, self.d_ffn, self.d_model)
        self.dropout4 = nn.Dropout(self.dropout,p=self.dropout)
        self.norm3 = nn.LayerNorm([self.d_model])

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> MLFusion:  # noqa: F821
        return cls(**config)

    @staticmethod
    def _get_activation_fn(name: str, *args, **kwargs) -> nn.Cell:
        """Returns an activation function instance.

        Arguments:
            name: Name of the activation function module.

        Returns:
            Activation function module instance.
        """
        return getattr(nn, name)(*args, **kwargs)

    @staticmethod
    def _get_ffn_layer(name: str, *args, **kwargs) -> nn.Cell:
        """Returns an feed forward network layer instance.

        Arguments:
            name: Name of the feed forward layer module.

        Retruns:
            Feed forward network module instance.
        """
        try:
            return getattr(layers, name)(*args, **kwargs)
        except AttributeError:
            return getattr(nn, name)(*args, **kwargs)
        except Exception as e:
            raise e

    @staticmethod
    def with_pos_embed(tensor: ms.Tensor, pos: ms.Tensor = None) -> ms.Tensor:
        """Returns a positional embedded tensor.

        Arguments:
            tensor: A tensor with shape (B, N, C)
            pos: A positional embedding with shape (B, N, C)

        Retruns:
            Positional embedded tensor.
        """
        return tensor if pos is None else tensor + pos

    def forward_self_attn(self,
                          query: ms.Tensor,
                          query_positions: ms.Tensor = None) -> ms.Tensor:
        """Returns the self attended query features.

        Arguments:
            query: Query feature tensor with shape (B, N, d_model).
            query_positions: Positional embedding values of the query features
                with shape (B, N, d_model)

        Returns:
            out: Output feature tensor with shape (B, N, d_model).
        """
        # Apply positional embedding
        q = k = self.with_pos_embed(query, query_positions)

        # Apply self attention
        out = self.self_attn(q, k, query)[0]

        # Apply dropout
        out = query + self.dropout1(out)

        # Apply normalization
        if self.norm:
            out = self.norm1(out)

        return out

    def forward_cross_attn(self,
                           query: ms.Tensor,
                           batch: Dict[str, ms.Tensor],
                           reference_points: ms.Tensor,
                           query_positions: ms.Tensor = None) -> ms.Tensor:
        """Returns query features based on the given batch feature levels and reference points.

        Arguments:
            query: A tensor of query featurs used during the attention with
                shape (B, N, d_model).
            batch: Dictionary of multi-level input features with length n_levels.
                The input batch represent the keys and values to attend to.
                The tensors are of shape (B, H, W, d_model)
            reference_points: A tensor representing normalized reference points
                with shape (B, N, 2).
            query_positions: Positional embedding values of the query features
                with shape (B, N, d_model)

        Returns:
            out: Fused multi-level output features with shape (B, N, d_model).
        """
        # Get input feature map dimensions
        input_spatial_shapes = ops.stack(
            tuple((ms.Tensor(l.shape[1:3]) for l in batch.values())),
            axis=0
        )
        input_spatial_shapes = ops.atleast_2d(input_spatial_shapes)

        # Flatten input features
        input_flatten = ops.cat(tuple((self.flatten1(l) for l in batch.values())), axis=1)

        # Determine flattend feature start indices
        input_level_start_index = ops.cumsum(
            ms.Tensor([0] + [l.shape[1] * l.shape[2] for l in batch.values()]),axis=0)[:-1]

        # Repeat reference points for each level
        # ref_points = reference_points.unsqueeze(2).repeat(1, 1, len(batch), 1)
        ref_points = ops.tile(reference_points.unsqueeze(2),(1, 1, len(batch), 1))

        # Query features from the current view
        out = self.ms_deform_attn(
                self.with_pos_embed(query, query_positions),
                ref_points,
                input_flatten,
                input_spatial_shapes,
                input_level_start_index
        )

        # Apply dropout
        out = query + self.dropout2(out)

        # Apply normalization
        if self.norm:
            out = self.norm2(out)

        return out

    def forward_ffn(self, query: ms.Tensor) -> ms.Tensor:
        """Returns refined query features.

        Arguments:
            query: Query feature tensor with shape (B, N, d_model).

        Returns:
            out: Output feature tensor with shape (B, N, d_model).
        """
        # Apply feed forward layers
        out = self.ffn2(self.dropout3(self.activation1(self.ffn1(query))))

        # Apply dropout
        out = query + self.dropout4(out)

        # Apply normalization
        if self.norm:
            out = self.norm3(out)

        return out

    def construct(self,
                query: ms.Tensor,
                batch: Dict[str, ms.Tensor],
                reference_points: ms.Tensor,
                query_positions: ms.Tensor = None) -> ms.Tensor:
        """Returns query features based on the given input and reference points.

        Arguments:
            batch: Dictionary of multi-level input features with length n_levels.
                The input batch represent the keys and values to attend to.
                The tensors are of shape (B, H, W, d_model)
            query: A tensor of query featurs used during the attention with
                shape (B, N, d_model).
            reference_points: A tensor representing normalized reference points
                with shape (B, N, 2).
            query_positions: Positional embedding values of the query features
                with shape (B, N, d_model)

        Returns:
            out: Fused multi-level output features with shape (B, N, d_model).
        """
        # Self attention: Self attend to the queries
        out = self.forward_self_attn(query=query, query_positions=query_positions)

        # Cross attention: Cross attend to multi level features
        out = self.forward_cross_attn(query=out, batch=batch,
                                      reference_points=reference_points,
                                      query_positions=query_positions)

        # FFN: Propagate attended features
        out = self.forward_ffn(query=out)


        return out


class MPFusion(nn.Cell):
    def __init__(self,
                 m_views: int,
                 d_model: int = 256,
                 d_ffn: int = 1024,
                 n_levels: List[int] = None,
                 n_heads: List[int] = None,
                 n_points: List[int] = None,
                 ffn_layer: str = 'Linear',
                 activation: str = 'ReLU',
                 dropout: float = 0.0,
                 norm: bool = False,
                 reduction: str = 'mean',
                 **kwargs):
        """Multi-Perspective Fusion Transformer.

        Arguments:
            m_views: Number of perspective views to query from.
            d_model: Hidden feature (channel) dimension.
            n_levels: Number of feature levels for each view.
            n_heads: Number of attention heads for each view.
            n_points: Number of sampling points per attention head and
                per feature level for each view.
            reduction: Reduction mode to fuse the queries of multiple views.
                One of either mean, max, unary, linear, cross-attn or ffn.
        """
        # Initialize base class
        super().__init__()

        # Check input arguments
        if reduction not in {'mean', 'max', 'unary', 'linear', 'cross-attn', 'ffn'}:
            raise ValueError(
                f"The reduction mode must be one of either "
                f"'mean', 'max', 'unary', 'linear', 'cross-attn' or 'ffn' but {reduction} "
                f"was given!"
            )

        # Initialize instance attributes
        self.m_views = m_views
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.n_levels = n_levels if n_levels is not None else [1] * m_views
        self.n_heads = n_heads if n_heads is not None else [1] * m_views
        self.n_points = n_points if n_points is not None else [1] * m_views
        self.ffn_layer = ffn_layer
        self.activation = activation
        self.dropout = dropout
        self.norm = norm
        self.reduction = reduction

        # Initialize module layers (one for each view)
        self.ml_fusion_layers = nn.CellDict({
            'ms_deform_attn' + str(v):
            MLFusion(self.d_model, self.d_ffn, l, h, p,
                     self.ffn_layer, self.activation, self.dropout, self.norm)
            for v, l, h, p in zip(range(self.m_views), self.n_levels, self.n_heads, self.n_points)
        })

        # Initialize reduction (fusion) layer
        self.reduction_layer = self._init_reduction(self.reduction)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> MPFusion:  # noqa: F821
        return cls(**config)

    @staticmethod
    def with_pos_embed(tensor: ms.Tensor, pos: ms.Tensor = None) -> ms.Tensor:
        """Returns a positional embedded tensor.

        Arguments:
            tensor: A tensor with shape (B, N, C)
            pos: A positional embedding with shape (B, N, C)

        Retruns:
            Positional embedded tensor.
        """
        return tensor if pos is None else tensor + pos

    @staticmethod
    def _get_activation_fn(name: str, *args, **kwargs) -> nn.Cell:
        """Returns an activation function instance.

        Arguments:
            name: Name of the activation function module.

        Returns:
            Activation function module instance.
        """
        return getattr(nn, name)(*args, **kwargs)

    @staticmethod
    def _get_ffn_layer(name: str, *args, **kwargs) -> nn.Cell:
        """Returns an feed forward network layer instance.

        Arguments:
            name: Name of the feed forward layer module.

        Retruns:
            Feed forward network module instance.
        """
        try:
            return getattr(layers, name)(*args, **kwargs)
        except AttributeError:
            return getattr(nn, name)(*args, **kwargs)
        except Exception as e:
            raise e

    def _init_reduction(self, reduction: str) -> Union[Callable, nn.Cell]:
        """Returns a reduction module instance.

        Arguments:
            reduction: Selected reduction mode.

        Returns:
            Reduction module instance.
        """
        if reduction == 'mean':
            return partial(ms.mean, dim=-1)

        if reduction == 'max':
            return partial(ms.max, dim=-1)

        if reduction == 'unary':
            return layers.Unary1d(in_channels=self.m_views * self.d_model,
                                  out_channels=self.d_model,
                                  bias=False, channels_last=True)

        if reduction == 'linear':
            return nn.Linear(in_features=self.m_views * self.d_model, out_features=self.d_model,
                             bias=False)

        if reduction == 'cross-attn':
            return nn.MultiheadAttention(self.d_model, min(self.n_heads), dropout=self.dropout,
                                         kdim=self.d_model * self.m_views,
                                         vdim=self.d_model * self.m_views, batch_first=True)

        if reduction == 'ffn':
            return nn.CellDict({
                'ffn1': self._get_ffn_layer(self.ffn_layer, self.m_views * self.d_model,
                                            self.m_views * self.d_model),
                'activation1': self._get_activation_fn(self.activation),
                'dropout1': nn.Dropout(self.dropout),
                'ffn2': self._get_ffn_layer(self.ffn_layer, self.m_views * self.d_model,
                                            self.d_model),
                'downsample1': self._get_ffn_layer(self.ffn_layer, self.m_views * self.d_model,
                                                   self.d_model),
                'dropout2': nn.Dropout(self.dropout),
                'norm1': nn.LayerNorm(self.d_model),
            })

    def reduce(self,
               query: ms.Tensor,
               queries: ms.Tensor,
               query_positions: ms.Tensor) -> ms.Tensor:
        """Applies the selected reduction to the queries.

        Arguments:
            query: Original input query with shape (B, N, d_model)
            queries: Multi-view queries with shape (B, N, d_model, m_views)
            query_positions: Positional embedding values of the query features
                with shape (B, N, d_model)

        Returns:
            Reduced (fused) queries with shape (B, N, d_model)
        """
        if self.reduction in {'mean', 'max'}:
            return self.reduction_layer(queries)

        if self.reduction in {'unary', 'linear'}:
            # Get query dimensions
            B, N = query.shape[:2]

            return self.reduction_layer(queries.view(B, N, self.d_model * self.m_views))

        if self.reduction == 'cross-attn':
            # Get query dimensions
            B, N = query.shape[:2]

            return self.reduction_layer(
                query=self.with_pos_embed(query, query_positions),
                key=queries.view(B, N, self.d_model * self.m_views),
                value=queries.view(B, N, self.d_model * self.m_views),
                need_weights=False)[0]

        if self.reduction == 'ffn':
            # Get query dimensions
            B, N = query.shape[:2]

            queries = queries.view(B, N, self.d_model * self.m_views)

            # Apply ffn (similar to residual block)
            out = self.reduction_layer['ffn1'](queries)
            out = self.reduction_layer['activation1'](out)
            out = self.reduction_layer['dropout1'](out)
            out = self.reduction_layer['ffn2'](out)
            out = self.reduction_layer['dropout2'](out)

            queries = self.reduction_layer['downsample1'](queries)

            out = queries + out

            if self.norm:
                out = self.reduction_layer['norm1'](out)

            return out

    def construct(self,
                query: ms.Tensor,
                batch: List[Dict[str, ms.Tensor]],
                reference_points: List[ms.Tensor],
                query_positions: ms.Tensor) -> ms.Tensor:
        """Returns fused query features based on the given input and reference points.

        Arguments:
            batch: List of ordered dictionaries mapping a level to a tensor.
                The list has length m_views and each dict has length n_levels.
                The input batch represent the keys and values to attend to.
                The tensors are of shape (B, H, W, d_model)
            query: A tensor of query featurs used during the attention with
                shape (B, N, d_model).
            reference_points: A list of tensors representing normalized
                reference points. The list has length m_views and the tensors
                have shape (B, N, 2).
            query_positions: Positional embedding values of the query features
                with shape (B, N, d_model)

        Returns:
            out: Fused output features with shape (B, N, d_model).
        """
        # Initialize queries with shape (query, m_views)
        queries = ops.zeros(query.shape + (self.m_views, ), dtype=query.dtype)
        # Define iterator
        iterator = zip(self.ml_fusion_layers.values(), batch, reference_points)

        for i, (ml_fusion_layer, levels, ref_points) in enumerate(iterator):
            # Query features from the current view
            queries[..., i] = ml_fusion_layer(
                    query,
                    levels,
                    ref_points,
                    query_positions
            )
        # Fuse multi perspective query features
        out = self.reduce(query, queries, query_positions)


        return out


class IMPFusion(nn.Cell):
    def __init__(self,
                 i_iter: int = 1,
                 m_views: int = 1,
                 d_model: int = 256,
                 d_ffn: int = 1024,
                 n_queries: int = 100,
                 n_levels: List[int] = None,
                 n_heads: List[int] = None,
                 n_points: List[int] = None,
                 q_init: str = 'uniform_',
                 ffn_layer: str = 'Linear',
                 activation: str = 'ReLU',
                 dropout: float = 0.0,
                 norm: bool = False,
                 reduction: str = 'mean',
                 head: nn.Cell = None,
                 **kwargs):
        """Iterative Multi-Perspective Fusion Transformer.

        Arguments:
            i_iter: Number of fusion and output refinement iterations.
            m_views: Number of perspective views to query from.
            d_model: Hidden feature (channel) dimension.
            n_queries: Number of queries fed to the fuser.
            n_levels: Number of feature levels for each view.
            n_heads: Number of attention heads for each view.
            n_points: Number of sampling points per attention head and
                per feature level for each view.
            q_init: Query feature initialization method.
            reduction: Reduction mode to fuse the queries of multiple views.
                One of either mean, max or cross-attn.
            head: Output head of the model. Used to generate the input
                for the next iteration and the final output.
        """
        # Initialize base class
        super().__init__()

        # Initialize instance attributes
        self.i_iter = i_iter
        self.m_views = m_views
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.n_queries = n_queries
        self.n_levels = n_levels if n_levels is not None else [1] * m_views
        self.n_heads = n_heads if n_heads is not None else [1] * m_views
        self.n_points = n_points if n_points is not None else [1] * m_views
        self.ffn_layer = ffn_layer
        self.activation = activation
        self.dropout = dropout
        self.norm = norm
        self.reduction = reduction

        initializer = ms.common.initializer.Uniform(scale=0.1) 
        self.q_init = ms.Tensor(shape=(self.n_queries,self.d_model), dtype=ms.float32, init=initializer)

        if head is None:
            head = nn.Identity()

        # Initialize fusion layers
        self.mpfusion = nn.CellDict({
            'fusion' + str(i):
            MPFusion(self.m_views, self.d_model, self.d_ffn, self.n_levels,
                     self.n_heads, self.n_points, self.ffn_layer, self.activation,
                     self.dropout, self.norm, self.reduction)
            for i in range(self.i_iter)
        })

        # Initialize detection heads
        self.heads = self._get_clones(head, self.i_iter)

        # Initialize query positional embedding
        self.query_embedding = nn.Embedding(self.n_queries, self.d_model)

        # Initialize queries
        query = ops.zeros((self.n_queries, self.d_model))
        self.query = ms.Parameter(query)

        self.reset_parameters()

    @classmethod
    def from_config(cls, config: Dict[str, Any], **kwargs) -> IMPFusion:  # noqa: F821
        return cls(**config, **kwargs)

    @staticmethod
    def _get_clones(module: nn.Cell, n: int) -> nn.Cell:
        """Retruns a module list of n cloned modules.

        Arguments:
            module: Modules to clone.
            n: Number of clones.

        Returns:
            A module list of n clones of the given module.
        """
        return nn.CellList([deepcopy(module) for i in range(n)])

    def reset_parameters(self) -> None:
        """Initialize query weights."""
        pass

    def get_reference_points(self,
                             query: ms.Tensor,
                             transformation: ms.Tensor,
                             projection: ms.Tensor,
                             shape: ms.Size) -> ms.Tensor:
        """Returns the query reference points in the feature space given a projection.

        Projects the query points (X, Y, Z) to the feature space (u, v)
        given a (4, 4) transformation matrix.

        [uw]   [p11 p12 p13 p14] [X]
        [vw] = [p21 p22 p23 p24] [Y]
        [ w]   [p31 p32 p33 p34] [Z]
        [ 1]   [0   0   0   1  ] [1]

        Arguments:
            query:  Query points with shape (B, N, 3).
            transformation: Transformation matrx given as (4, 4) homogeneous
                transformation matrix. If a transformation matrix is provided,
                the transformation is applied first (in cartesian space) before
                the reference points are transformed into spherical coordinates.
            projection: Projection matrx given as (4, 4) homogeneous
                transformation matrix. Projects the reference points into
                the sensor space.
            shape: Feature map shape with shape (B, 2).

        Returns:
            reference_points: Reference points with shape
                (B, N, 2) where 2 is ordered by H, W
        """
        if transformation.any():
            # Apply transformation to query points (T @ Q^T)
            # reference_points = ms.einsum(
            #     'bij,bkj->bki',
            #     transformation,
            #     ms.dstack((query[..., :3], ms.ones_like(query[..., 0])))
            # )
            # 手动实现 einsum('bij,bkj->bki', transformation, query_t)

            query_t=ops.dstack((query[...,:3],ops.ones_like(query[...,0])))
            transformation_expanded = transformation.unsqueeze(1)  # (1, 1, 4, 4)
            query_t_expanded = query_t.unsqueeze(-1)               # (1, 400, 4, 1)
            result = transformation_expanded @ query_t_expanded     # (1, 400, 4, 1)
            reference_points = result.squeeze(-1)                   # (1, 400, 4)
                    


            # Convert reference points from cartesian to spherical coordinates
            r, phi, roh = cart2spher(
                reference_points[..., 0].asnumpy(),
                reference_points[..., 1].asnumpy(),
                reference_points[..., 2].asnumpy()
            )

            reference_points = ops.dstack((r, phi, roh))

        else:
            reference_points = query


        # Get reference points in the feature map space (P @ Q^T)
        # reference_points =ops.einsum(
        #     'bij,bkj->bki',
        #     projection,
        #     ops.dstack((reference_points[..., :3], ops.ones_like(reference_points[..., 0])))
        # )

        # 手动实现 einsum('bij,bkj->bki', projection, reference_points_t)
        reference_points_t=ops.dstack((reference_points[...,:3],ops.ones_like(reference_points[...,0])))
        projection_expanded = projection.unsqueeze(1)                    # (1, 1, 4, 4)
        reference_points_t_expanded = reference_points_t.unsqueeze(-1)   # (1, 400, 4, 1)
        result = projection_expanded @ reference_points_t_expanded       # (1, 400, 4, 1)
        reference_points = result.squeeze(-1)                            # (1, 400, 4)

        # Scale reference points with the w value
        mask = (reference_points[..., 2] != 0)

        # Width index (pixel)
        u = reference_points[..., 0]
        u[mask] = reference_points[..., 0][mask] / reference_points[..., 2][mask]

        # Height index (pixel)
        v = reference_points[..., 1]
        v[mask] = reference_points[..., 1][mask] / reference_points[..., 2][mask]

        # Scale reference points according to the feature map size
        u = (u - 0) / (shape[:, 1].unsqueeze(1) - 0) * (1 - 0) + 0
        v = (v - 0) / (shape[:, 0].unsqueeze(1) - 0) * (1 - 0) + 0

        # Reduce reference points to 2d projection
        reference_points = ops.dstack((u, v))

        # Clip values to account for numerical issues
        reference_points = ops.clip(reference_points, min=0.0, max=1.0)

        return reference_points

    def construct(self,
                batch: List[Dict[str, ms.Tensor]],
                shape: List[ms.Tensor],
                projection: List[Tuple[ms.Tensor, ms.Tensor]],
                out: Dict[str, ms.Tensor]) -> ms.Tensor:
        """Returns an iteratively fused and refined output prediction.

        Arguments:
            batch: List of ordered dictionaries mapping a level to a tensor.
                The list has length m_views and each dict has length n_levels.
            shape: List of tensors representing the raw data input shapes.
                The list has length m_views and the tensors have shape (B, 2).
            projection: List of tuples, each containing two tensors representing
                homogeneous transformation matrices with shape (4, 4). The list
                has length m_views.
            out: Ordered dictionary of tensors that contains at least this entry:
                "center": Bounding box center coordinates of shape (B, N, 3).

        Returns:
            out: Ordered dictionary of tensors that contains these entries:
                "class": Bounding box class probabilities of shape (B, N, num_classes)
                "center": Bounding box center coordinates of shape (B, N, 3).
                "size": Bounding box size values of shape (B, N, 3).
                "angle": Bounding box orientation values of shape (B, N, 2).
        """
        # Get batch size
        B = out['center'].shape[0]

        # Adjust query dimensions (N, d_model) -> (B, N, d_model)
        query = ops.tile(self.query.unsqueeze(0),(B, 1, 1))

        # Get query positional embedding values (B, N, d_model)
        query_pos = ops.tile(self.query_embedding.embedding_table.unsqueeze(0),(B, 1, 1))
        i=0
        for layer, head in zip(self.mpfusion.values(), self.heads):
            # Calculate reference points for each feature map   
            reference_points = [
                self.get_reference_points(out['center'][..., :3], p[0], p[1], s)
                for p, s in zip(projection, shape)
            ]
            i+=1
            # Query features from multiple perspectives
            query = layer(query, batch, reference_points, query_pos)
            # Apply head to query features
            out = head(query, out)

        return out


def build_mpfusion(*args, **kwargs):
    return IMPFusion.from_config(*args, **kwargs)


def cart2spher(x: np.ndarray,
               y: np.ndarray,
               z: np.ndarray,
               degrees: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert Cartesian coordinates to spherical coordinates (elevation version)

    Conventions:
    - r: Radius (distance to the origin)
    - phi: Azimuthal angle, the angle between the x-axis and the y-z plane, range [-180°, 180°] or [-π, π]
    - roh: Elevation angle, the angle between the x-y plane and the z-axis, range [-90°, 90°] or [-π/2, π/2]

    Parameters:
        x: Array of x-coordinates
        y: Array of y-coordinates
        z: Array of z-coordinates
        degrees: Whether to return angles in degrees (True) or radians (False)

    Returns:
        r: Radius
        phi: Azimuthal angle
        roh: Elevation angle
    """

    r = np.linalg.norm(np.stack([x, y, z], axis=-1), axis=-1)
    r = r.reshape(x.shape)
    
    phi = np.arctan2(y, x)
    
    mask = (r != 0)
    c = np.zeros_like(z)
    c[mask] = z[mask] / r[mask]
    roh = np.arcsin(c)
    
    if degrees:
        phi = np.rad2deg(phi)
        roh = np.rad2deg(roh)
    
    r = ms.Tensor(r)
    phi = ms.Tensor(phi)
    roh = ms.Tensor(roh)
    return r, phi, roh



