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
"""Utils"""
from typing import Any

import numpy as np
from mindspore import Tensor

FP16_RTOL = 1e-3
FP16_ATOL = 1e-3
FP32_RTOL = 1e-4
FP32_ATOL = 1e-4


def compare_output(output_1, output_2, rtol=1e-5, atol=1e-5):
    r"""
    Compares model outputs and determines if they match within the specified tolerance
    Args:
        output_1 (Union[np.ndarray, Tuple[np.ndarray, ...]]): First model output to compare
        output_2 (Union[np.ndarray, Tuple[np.ndarray, ...]]): Second model output to compare
        rtol (float): Relative tolerance for allowed error, default is 1e-5
        atol (float): Absolute tolerance for allowed error, default is 1e-5

    Returns:
        bool: Whether the outputs match within the given tolerance
    """
    # Output of tensor
    if isinstance(output_1, np.ndarray):
        return np.allclose(output_1, output_2, rtol, atol, equal_nan=True)
    # Output of tuple of tensors
    if isinstance(output_1, tuple):
        # Loop through tuple of outputs
        for _, (out_1, out_2) in enumerate(zip(output_1, output_2)):
            # If tensor use allclose
            if isinstance(out_1, np.ndarray):
                if not np.allclose(out_1, out_2, rtol, atol, equal_nan=True):
                    return False
            # Otherwise assume primitive
            else:
                if not out_1 == out_2:
                    return False
    # Unsupported output type
    else:
        print(
            "Model returned invalid type for unit test, should be np.ndarray or Tuple[np.ndarray]"
        )
        return False

    return True


def tensor_to_numpy(data: Any) -> Any:
    """Convert MindSpore Tensors to NumPy arrays recursively.
    This function traverses the input data structure and converts all MindSpore Tensors
    to NumPy arrays, while leaving other data types unchanged.
    Args:
        data (Any): Input data which can be a MindSpore Tensor, dict, list, tuple, or other types.
    Returns:
        Any: Data structure with MindSpore Tensors converted to NumPy arrays.
    """
    if isinstance(data, Tensor):
        return data.numpy()
    if isinstance(data, dict):
        return {k: tensor_to_numpy(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return type(data)(tensor_to_numpy(v) for v in data)
    return data


def numpy_to_tensor(data: Any) -> Any:
    """Convert NumPy arrays to MindSpore Tensors recursively.
    This function traverses the input data structure and converts all NumPy arrays
    to MindSpore Tensors, while leaving other data types unchanged.
    Args:
        data (Any): Input data which can be a NumPy array, dict, list, tuple, or other types.
    Returns:
        Any: Data structure with NumPy arrays converted to MindSpore Tensors.
    """
    if isinstance(data, np.ndarray):
        return Tensor(data)
    if isinstance(data, dict):
        return {k: numpy_to_tensor(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return type(data)(numpy_to_tensor(v) for v in data)
    return data


def is_equal(a: Any, b: Any) -> bool:
    """Compare two objects for equality with special handling for different types.

    This function performs a deep comparison between two objects, supporting:
    - NumPy arrays (using tolerance-based comparison)
    - Dictionaries (recursive comparison of values)
    - Lists and tuples (element-wise comparison)
    - NamedTuples (field-wise comparison)
    - Other types (using standard equality comparison)

    Args:
        a (Any): First object to compare
        b (Any): Second object to compare

    Returns:
        bool: True if objects are considered equal, False otherwise

    Examples:
        >>> is_equal(np.array([1.0]), np.array([1.0]))
        True
        >>> is_equal({'a': 1, 'b': 2}, {'a': 1, 'b': 2})
        True
        >>> is_equal([1, 2, 3], [1, 2, 3])
        True
    """
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return compare_output(a, b, FP32_ATOL, FP32_RTOL)
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return False
        return all(is_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(is_equal(x, y) for x, y in zip(a, b))
    if hasattr(a, "_fields") and hasattr(b, "_fields"):  # NamedTuple 支持
        if a._fields != b._fields:
            return False
        return all(is_equal(getattr(a, f), getattr(b, f)) for f in a._fields)
    return a == b
