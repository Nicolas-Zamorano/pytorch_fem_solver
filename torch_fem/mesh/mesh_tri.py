"""Class for triangular mesh representation"""

from typing import Tuple
import torch
from .abstract_mesh import AbstractMesh
import tensordict


class MeshTri(AbstractMesh):
    """Class for triangular mesh representation"""

    @property
    def edges_permutations(self):
        return torch.tensor([[0, 1], [1, 2], [0, 2]])

    def _compute_vertices_4_edges(self, triangulation):

        vertices_4_edges = triangulation["edges", "vertices"]
        markers_4_edges = triangulation["edges", "markers"].squeeze(-1)

        vertices_4_boundary_edges = vertices_4_edges[markers_4_edges == 1]

        vertices_4_interior_edges = vertices_4_edges[markers_4_edges != 1]

        return vertices_4_boundary_edges, vertices_4_interior_edges

    def _compute_cells_4_edges(
        self, triangulation, vertices_4_boundary_edges, vertices_4_interior_edges
    ):

        # if "neighbors" in triangulation["cells"]:
        #     neighbors = triangulation["cells", "neighbors"]

        #     number_neighbors = neighbors.shape[-2]

        #     cells_idx = torch.arange(number_neighbors).repeat_interleave(
        #         triangulation["cells", "vertices"].shape[-1]
        #     )

        #     neigh_flat = neighbors.reshape(-1)

        #     mask_inner = neigh_flat != -1
        #     mask_boundary = neigh_flat == -1

        #     tri1 = cells_idx[mask_inner]
        #     tri2 = neigh_flat[mask_inner]

        #     pair = torch.stack(
        #         [torch.minimum(tri1, tri2), torch.maximum(tri1, tri2)], dim=1
        #     )

        #     cells_4_interior_edges = torch.unique(pair, dim=0)

        #     cells_4_boundary_edges = cells_idx[mask_boundary]

        # else:
        vertices_4_cells = triangulation["cells", "vertices"]

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
        )[1].reshape(-1, 2)

        return cells_4_boundary_edges, cells_4_interior_edges

    def _compute_edges_vertices(self, triangulation):
        vertices = triangulation["cells", "vertices"]
        vertices_4_edges = vertices[..., self.edges_permutations.to(vertices.device)]
        edges_flat = vertices_4_edges.reshape(-1, 2)

        # Count occurrences per undirected edge key
        edges_list = edges_flat.tolist()
        counts = {}
        for a, b in edges_list:
            key = (a, b) if a <= b else (b, a)
            counts[key] = counts.get(key, 0) + 1

        # Keep first appearance and preserve its orientation; mark boundary if count == 1
        seen = set()
        unique_edges_list = []
        boundary_mask_list = []
        for a, b in edges_list:
            key = (a, b) if a <= b else (b, a)
            if key in seen:
                continue
            seen.add(key)
            unique_edges_list.append([a, b])  # preserve original orientation
            boundary_mask_list.append(1 if counts[key] == 1 else 0)

        unique_edges = torch.tensor(
            unique_edges_list, dtype=edges_flat.dtype, device=edges_flat.device
        )
        keep_mask = torch.tensor(
            boundary_mask_list, dtype=torch.long, device=edges_flat.device
        ).unsqueeze(-1)

        return unique_edges, keep_mask

    def _compute_cells_max_length(self, triangulation):
        vertices_4_edges, _ = torch.sort(
            triangulation["cells", "vertices"][..., self.edges_permutations], dim=-1
        )

        coordinates_4_edges = self.compute_coordinates_4_cells(
            triangulation["vertices", "coordinates"], vertices_4_edges
        )

        coordinates_4_edges_first_vertex, coordinates_4_edges_second_vertex = (
            torch.split(coordinates_4_edges, 1, dim=-2)
        )

        diameter_4_cells = torch.max(
            torch.norm(
                coordinates_4_edges_second_vertex - coordinates_4_edges_first_vertex,
                dim=-1,
            ),
            dim=-2,
            keepdim=False,
        )[0]

        return diameter_4_cells

    def _compute_interior_edge_lengths_and_normals(
        self,
        coordinates_4_interior_edges,
        cells_4_interior_edges,
        coordinates_4_cells,
    ):

        (
            coordinates_4_interior_edges_first_vertex,
            coordinates_4_interior_edges_second_vertex,
        ) = torch.split(coordinates_4_interior_edges, 1, dim=-2)

        interior_edges_vector = (
            coordinates_4_interior_edges_second_vertex
            - coordinates_4_interior_edges_first_vertex
        )

        interior_edges_length = torch.norm(interior_edges_vector, dim=-1, keepdim=True)

        normal_4_interior_edges = (
            interior_edges_vector[..., [1, 0]]
            * torch.tensor([-1.0, 1.0])
            / interior_edges_length
        )

        # Fix orientation

        centroids_4_cells_4_interior_edges = self.compute_coordinates_4_cells(
            coordinates_4_cells, cells_4_interior_edges
        ).mean(dim=-2)

        (
            first_centroid_4_interior_edges,
            second_centroid_4_interior_edges,
        ) = torch.split(centroids_4_cells_4_interior_edges, 1, dim=-2)

        normal_direction_4_interior_edges = (
            normal_4_interior_edges
            * (second_centroid_4_interior_edges - first_centroid_4_interior_edges)
        ).sum(dim=-1)

        normal_4_interior_edges = torch.where(
            normal_direction_4_interior_edges[..., None] < 0,
            -normal_4_interior_edges,
            normal_4_interior_edges,
        )

        return interior_edges_length, normal_4_interior_edges
