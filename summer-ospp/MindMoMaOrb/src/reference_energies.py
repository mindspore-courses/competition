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
"""Reference Energies."""

from typing import NamedTuple
import os
import json

import numpy


class ReferenceEnergies(NamedTuple):
    """
    Reference energies for an atomic system.

    Our vasp reference energies are computed by running vasp
    optimisations on a single atom of each atom-type.

    Other reference energies are fitted using least-squares.

    Doing so with mp-traj-d3 gives the following:

    ---------- LSTQ ----------
    Reference MAE:  13.35608855004781
    (energy - ref) mean: 1.3931169304958624
    (energy - ref) std: 22.45615276341948
    (energy - ref)/natoms mean: 0.16737045963056316
    (energy - ref)/natoms std: 0.8189314920219992
    CO2: Predicted vs DFT: -23.154158610392408 vs -22.97
    H2O: Predicted vs DFT: -11.020918107591324 vs - 14.23
    ---------- VASP ----------
    Reference MAE:  152.4722089438871
    (energy - ref) mean: -152.47090833346033
    (energy - ref) std: 153.89049784836962
    (energy - ref)/natoms mean: -4.734136414817941
    (energy - ref)/natoms std: 1.3603868419157275
    CO2: Predicted vs DFT: -4.35888857 vs -22.97
    H2O: Predicted vs DFT: -2.66521147 vs - 14.23
    ---------- Shifted VASP ----------
    Reference MAE:  28.95948216608197
    (energy - ref) mean: 0.7083632520428979
    (energy - ref) std: 48.61861182844561
    (energy - ref)/natoms mean: 0.17320099403091083
    (energy - ref)/natoms std: 1.3603868419157275
    CO2: Predicted vs DFT: -19.080900796546562 vs -22.97
    H2O: Predicted vs DFT: -12.479886287697706 vs - 14.23

    Args:
        coefficients: Coefficients for each atom in the periodic table.
            Must be of length 118 with first entry equal to 0.
        residual_mean: Mean of (pred - target)
        residual_std: Standard deviation of (pred - target)
        residual_mean_per_atom: Mean of (pred - target)/n_atoms.
        residual_std_per_atom: Standard deviation of (pred - target)/n_atoms.
    """

    coefficients: numpy.ndarray
    residual_mean: float
    residual_std: float
    residual_mean_per_atom: float
    residual_std_per_atom: float


# NOTE: we have only computed these for the first
# 88 elements, and padded the remainder with 0.
def get_reference_energies(energy_type: str = None) -> ReferenceEnergies:
    """
    Get reference energies from JSON file.

    Args:
        energy_type: Type of energy reference ('vasp', 'vasp-shifted', or 'mp-traj-d3').
                    If None, returns all types.

    Returns:
        If energy_type is specified, returns a ReferenceEnergies object.
        If energy_type is None, returns a dict of all ReferenceEnergies objects.
    """
    json_path = os.path.join(os.path.dirname(__file__), 'reference_energies_data.json')
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Reference energies data file not found at {json_path}") from exc

    def create_reference_energy(energy_data):
        return ReferenceEnergies(
            coefficients=numpy.array(energy_data['coefficients']),
            residual_mean=energy_data['residual_mean'],
            residual_std=energy_data['residual_std'],
            residual_mean_per_atom=energy_data['residual_mean_per_atom'],
            residual_std_per_atom=energy_data['residual_std_per_atom']
        )

    if energy_type is not None:
        if energy_type not in data:
            raise ValueError(f"Unknown energy type: {energy_type}")
        return create_reference_energy(data[energy_type])

    return {
        k: create_reference_energy(v) for k, v in data.items()
    }
