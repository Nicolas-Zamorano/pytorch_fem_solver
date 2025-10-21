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

    def map_fine_mesh(self, fine_mesh: AbstractMesh) -> torch.Tensor:
        """
        Maps each triangle (element) of a fine mesh to the corresponding triangle in a coarse mesh.
        Given a fine mesh and the current (coarse) mesh, this method returns a tensor of shape
        (n_elements_fine,) where each entry i indicates the index of the triangle in the coarse mesh
        that contains the centroid of the i-th triangle in the fine mesh. If a fine mesh triangle's
        centroid is not contained in any coarse mesh triangle, its entry is set to -1.
        """
        c4e_finer_mesh = fine_mesh["cells", "coordinates"]  # (n_elem_h, 3, 2)
        c4e_coarser_mesh = self["cells", "coordinates"]  # (n_elem_H, 3, 2)
        centroids_finer_mesh = c4e_finer_mesh.mean(dim=-2)  # (n_elem_h, 2)

        # Expand for broadcasting
        P = centroids_finer_mesh.unsqueeze(1)  # (n_elem_h, 1, 2)
        A = c4e_coarser_mesh[:, 0, :].unsqueeze(0)  # (1, n_elem_H, 2)
        B = c4e_coarser_mesh[:, 1, :].unsqueeze(0)  # (1, n_elem_H, 2)
        C = c4e_coarser_mesh[:, 2, :].unsqueeze(0)  # (1, n_elem_H, 2)

        v0 = C - A  # (1, n_elem_H, 2)
        v1 = B - A  # (1, n_elem_H, 2)
        v2 = P - A  # (n_elem_h, n_elem_H, 2)

        dot00 = (v0 * v0).sum(dim=-1)  # (1, n_elem_H)
        dot01 = (v0 * v1).sum(dim=-1)  # (1, n_elem_H)
        dot11 = (v1 * v1).sum(dim=-1)  # (1, n_elem_H)

        dot02 = (v0 * v2).sum(dim=-1)  # (n_elem_h, n_elem_H)
        dot12 = (v1 * v2).sum(dim=-1)  # (n_elem_h, n_elem_H)

        denom = dot00 * dot11 - dot01 * dot01  # (1, n_elem_H)
        denom = denom.clamp(min=1e-14)

        u = (dot11 * dot02 - dot01 * dot12) / denom  # (n_elem_h, n_elem_H)
        v = (dot00 * dot12 - dot01 * dot02) / denom  # (n_elem_h, n_elem_H)

        inside = (u >= 0) & (v >= 0) & (u + v <= 1)  # (n_elem_h, n_elem_H)

        # Find first coarse triangle containing each fine triangle centroid
        mapping = torch.full((c4e_finer_mesh.shape[0],), -1, dtype=torch.long)

        # Use argmax to find first True value per row (or 0 if all False)
        has_match = inside.any(dim=1)
        first_match = inside.long().argmax(dim=1)

        # Only set mapping where there's actually a match
        mapping[has_match] = first_match[has_match]

        return mapping
