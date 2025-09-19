"""Class for handling multiple fractures represented as triangular meshes"""

import torch
from .meshes_tri import MeshesTri


class FracturesTri(MeshesTri):
    """Class for handling multiple fractures represented as triangular meshes"""

    def __init__(self, triangulations: list, fractures_3d_data: torch.Tensor):

        triangulations = self._stack_triangulations(triangulations)

        super().__init__(triangulations)

        self._compute_fracture_map(fractures_3d_data)

        self["vertices", "coordinates_3d"] = (
            self["jacobian_fracture_map"] @ self["vertices", "coordinates"].mT
            + self["translation_vector"]
        ).mT

        self["cells", "coordinates_3d"] = self.compute_coordinates_4_cells(
            self["vertices", "coordinates_3d"], self["cells", "vertices"]
        )

        self["interior_edges", "normals_3d"] = (
            self["jacobian_fracture_map"].unsqueeze(-3)
            @ self["interior_edges", "normals"].mT
            + self["translation_vector"].unsqueeze(-3)
        ).mT

    @property
    def _edges_permutations(self):
        return torch.tensor([[0, 1], [1, 2], [0, 2]])

    def _stack_triangulations(self, fracture_triangulations: list):
        """Stack multiple fracture triangulations into a single TensorDict"""

        fracture_triangulations_tensordict = [
            tensordict.TensorDict(fracture_triangulation)
            for fracture_triangulation in fracture_triangulations
        ]

        stacked_fractured_triangulations = torch.stack(
            fracture_triangulations_tensordict, dim=0
        )

        return stacked_fractured_triangulations

    def _compute_fracture_map(self, fractures_3d_data: torch.Tensor):
        """compute mapping for each fracture from the 2D space to 3D."""
        vertices_2d = self["vertices", "coordinates"][:, :3, :]

        vertices_3d = fractures_3d_data[:, :3, :]

        extended_vertices_2d = torch.cat(
            [vertices_2d, torch.ones_like(vertices_3d[..., [-1]])], dim=-1
        )

        linear_equation = vertices_3d.mT @ torch.inverse(extended_vertices_2d).mT

        jacobian_fracture_map = linear_equation[..., :2]
        translation_vector = linear_equation[..., [-1]]

        det_jacobian_fracture_map = torch.norm(
            torch.cross(*torch.split(jacobian_fracture_map, 1, dim=-1), dim=-2),
            dim=-2,
            keepdim=True,
        )

        inv_jacobian_fracture_map = (
            torch.inverse(jacobian_fracture_map.mT @ jacobian_fracture_map)
            @ jacobian_fracture_map.mT
        )

        self["jacobian_fracture_map"] = jacobian_fracture_map
        self["inv_jacobian_fracture_map"] = inv_jacobian_fracture_map
        self["det_jacobian_fracture_map"] = det_jacobian_fracture_map
        self["translation_vector"] = translation_vector
