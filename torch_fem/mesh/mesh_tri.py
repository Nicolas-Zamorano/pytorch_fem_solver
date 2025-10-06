"""Class for triangular mesh representation"""

import torch
from .abstract_mesh import AbstractMesh


class MeshTri(AbstractMesh):
    """Class for triangular mesh representation"""

    @property
    def edges_permutations(self):
        return torch.tensor([[0, 1], [1, 2], [0, 2]])
