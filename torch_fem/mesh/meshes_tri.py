"""Class to represent multiples triangular meshes"""

import torch
import tensordict
from .mesh_tri import MeshTri


class MeshesTri(MeshTri):
    """Class to represent multiples triangular meshes"""

    def __init__(self, triangulations: list):

        triangulations = self._stack_triangulations(triangulations)

        super().__init__(triangulations)

    def _stack_triangulations(
        self, fracture_triangulations: list
    ) -> tensordict.TensorDict:
        """Stack multiple fracture triangulations into a single TensorDict"""

        fracture_triangulations_tensordict = [
            tensordict.TensorDict(fracture_triangulation)
            for fracture_triangulation in fracture_triangulations
        ]

        stacked_fractured_triangulations = tensordict.stack(
            fracture_triangulations_tensordict, dim=0
        )

        return stacked_fractured_triangulations

    @staticmethod
    def compute_coordinates_4_cells(
        coordinates_4_vertices: torch.Tensor, vertices_4_cells: torch.Tensor
    ):
        """Compute the coordinates of the nodes in the mesh."""
        return coordinates_4_vertices[
            torch.arange(coordinates_4_vertices.size(0))[:, None, None],
            vertices_4_cells,
        ]

    @staticmethod
    def apply_mask(tensor: torch.Tensor, mask: torch.Tensor):
        """Compute masking for batched tensors"""
        return torch.cat(
            [
                batch_tensor[mask_batch].unsqueeze(0)
                for batch_tensor, mask_batch in zip(tensor, mask)
            ],
            dim=0,
        )

    def _compute_cells_4_edges(
        self,
        triangulation: tensordict.TensorDict,
        vertices_4_boundary_edges: torch.Tensor,
        vertices_4_interior_edges: torch.Tensor,
    ):
        if "neighbors" in triangulation["cells"]:
            neighbors = triangulation["cells", "neighbors"]
            number_meshes, number_cells, number_edges = neighbors.shape

            cells_idx = torch.arange(number_cells, device=neighbors.device)
            cells_idx = cells_idx.repeat_interleave(number_edges)
            cells_idx = cells_idx.unsqueeze(0).expand(number_meshes, -1)

            neigh_flat = neighbors.reshape(number_meshes, -1)

            mask_inner = neigh_flat != -1
            mask_boundary = neigh_flat == -1

            cells_4_interior_edges_list = []
            cells_4_boundary_edges_list = []

            cells_4_interior_edges = torch.tensor([])
            cells_4_boundary_edges = torch.tensor([])

            for m in range(number_meshes):

                mask_m = mask_inner[m]
                tri1_m = cells_idx[m][mask_m]
                tri2_m = neigh_flat[m][mask_m]

                pair_m = torch.stack(
                    [torch.minimum(tri1_m, tri2_m), torch.maximum(tri1_m, tri2_m)],
                    dim=1,
                )

                edges_int = torch.unique(pair_m, dim=0)
                edges_bou = cells_idx[m][mask_boundary[m]].unsqueeze(1)

                cells_4_interior_edges_list.append(edges_int)
                cells_4_boundary_edges_list.append(edges_bou)

                cells_4_interior_edges = torch.stack(cells_4_interior_edges_list, dim=0)
                cells_4_boundary_edges = torch.stack(cells_4_boundary_edges_list, dim=0)
        else:
            vertices_4_cells = triangulation["cells", "vertices"]

            number_meshes = vertices_4_cells.shape[0]

            cells_4_boundary_edges = (
                (
                    vertices_4_boundary_edges.unsqueeze(-2).unsqueeze(-2)
                    == vertices_4_cells.unsqueeze(-1).unsqueeze(-4)
                )
                .any(dim=-2)
                .all(dim=-1)
                .float()
                .argmax(dim=-1, keepdim=True)
            )
            cells_4_interior_edges = torch.nonzero(
                (
                    vertices_4_interior_edges.unsqueeze(-2).unsqueeze(-2)
                    == vertices_4_cells.unsqueeze(-1).unsqueeze(-4)
                )
                .any(dim=-2)
                .all(dim=-1),
                as_tuple=True,
            )[1].reshape(number_meshes, -1, 2)

        return cells_4_boundary_edges, cells_4_interior_edges

    def _compute_edges_vertices(self, triangulation: tensordict.TensorDict):
        """Compute vertices for unique edges."""

        vertices_4_edges_4_mesh = triangulation["cells", "vertices"][
            ..., self.edges_permutations
        ]

        vertices_4_unique_edges_list = []
        boundary_mask_list = []

        for vertices_4_edges in vertices_4_edges_4_mesh:

            vertices_4_unique_edges, _, boundary_mask = torch.unique(
                vertices_4_edges.reshape(-1, 2).mT,
                return_inverse=True,
                sorted=False,
                return_counts=True,
                dim=-1,
            )

            vertices_4_unique_edges_list.append(vertices_4_unique_edges.mT)
            boundary_mask_list.append(boundary_mask)

        vertices_4_unique_edges = torch.stack(vertices_4_unique_edges_list, dim=0)
        boundary_mask = torch.stack(boundary_mask_list, dim=0)

        return vertices_4_unique_edges, boundary_mask

    def _compute_vertices_4_edges(self, triangulation: tensordict.TensorDict):
        vertices_4_edges = triangulation["edges", "vertices"]
        markers_4_edges = triangulation["edges", "markers"].squeeze(-1)

        vertices_4_boundary_edges = self.apply_mask(
            vertices_4_edges, markers_4_edges == 1
        )

        vertices_4_interior_edges = self.apply_mask(
            vertices_4_edges, markers_4_edges != 1
        )

        return vertices_4_boundary_edges, vertices_4_interior_edges

    def _compute_cells_min_length(self, triangulation: tensordict.TensorDict):
        """For each cells, compute the smaller length of the edges."""
        vertices_4_edges, _ = torch.sort(
            triangulation["cells", "vertices"][..., self.edges_permutations], dim=-1
        )

        coordinates_4_edges = self.apply_mask(
            triangulation["vertices", "coordinates"], vertices_4_edges
        )

        coordinates_4_edges_first_vertex, coordinates_4_edges_second_vertex = (
            torch.split(coordinates_4_edges, 1, dim=-2)
        )

        diameter_4_cells = torch.min(
            torch.norm(
                coordinates_4_edges_second_vertex - coordinates_4_edges_first_vertex,
                dim=-1,
                keepdim=True,
            ),
            dim=-2,
            keepdim=True,
        )[0]

        return diameter_4_cells
