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
"""Calculator."""

from typing import Optional
from ase.calculators.calculator import Calculator

from src.atomic_system import SystemConfig, ase_atoms_to_atom_graphs
from src.graph_regressor import GraphRegressor

class ORBCalculator(Calculator):
    """Calculator for predicting properties of atomic systems using a GraphRegressor model.
    This calculator is designed to be used with ASE (Atomic Simulation Environment) and
    provides an interface for calculating properties such as energy, forces, and stress
    based on a trained graph neural network model.
    It supports both CPU and GPU execution, and can handle large atomic systems efficiently.
    """
    def __init__(
            self,
            model: GraphRegressor,
            brute_force_knn: Optional[bool] = None,
            system_config: SystemConfig = SystemConfig(radius=10.0, max_num_neighbors=20),
            **kwargs,
    ):
        """Initializes the calculator.

        Args:
            model (GraphRegressor): The finetuned model to use for predictions.
            brute_force_knn: whether to use a 'brute force' k-nearest neighbors method for graph construction.
                Defaults to None, in which case brute_force is used if a GPU is available (2-6x faster),
                but not on CPU (1.5x faster - 4x slower). For very large systems (>10k atoms),
                brute_force may OOM on GPU, so it is recommended to set to False in that case.
            system_config (SystemConfig): The config defining how an atomic system is featurized.
            **kwargs: Additional keyword arguments for parent Calculator class.
        """
        Calculator.__init__(self, **kwargs)
        self.results = {}  # type: ignore
        self.model = model
        self.system_config = system_config
        self.brute_force_knn = brute_force_knn

        # NOTE: we currently do not predict stress, but when we do,
        # we should add it here and also update calculate() below.
        properties = []
        if model.node_head is not None:
            properties += ["energy", "free_energy"]
        if model.graph_head is not None:
            properties += ["forces"]
        if model.stress_head is not None:
            properties += ["stress"]
        assert properties, "Model must have at least one output head."
        self.implemented_properties = properties

    def calculate(self, atoms=None):
        """Calculate properties.

        :param atoms: ase.Atoms object
        :return:
        """
        # call to base-class to set atoms attribute
        Calculator.calculate(self, atoms)

        # prepare data
        batch = ase_atoms_to_atom_graphs(
            atoms,
            system_config=self.system_config,
            brute_force_knn=self.brute_force_knn,
        )

        self.results = {}
        out = self.model.predict(batch)
        if "energy" in self.implemented_properties:
            self.results["energy"] = float(out["graph_pred"].detach().item())
            self.results["free_energy"] = self.results["energy"]

        if "forces" in self.implemented_properties:
            self.results["forces"] = out["node_pred"].detach().numpy()

        if "stress" in self.implemented_properties:
            raw_stress = out["stress_pred"].detach().numpy()
            # reshape from (1, 6) to (6,) if necessary
            self.results["stress"] = (
                raw_stress[0] if len(raw_stress.shape) > 1 else raw_stress
            )
