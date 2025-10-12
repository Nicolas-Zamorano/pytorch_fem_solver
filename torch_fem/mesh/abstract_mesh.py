"""Abstract class for mesh representation"""

import abc
from typing import Tuple, Any
import torch
import tensordict
from numpy import int32


class AbstractMesh(abc.ABC):
    """Abstract class for mesh representation"""

    def __init__(self, triangulation: dict[str, Any]):

        triangulation_tensordict = self._triangle_to_tensordict(triangulation)

        self._triangulation = self._build_optional_parameters(triangulation_tensordict)

    def __getitem__(
        self, key: str | Tuple[str, str]
    ) -> tensordict.TensorDict | torch.Tensor:
        return self._triangulation[key]

    def __setitem__(self, key: str, value: tensordict.TensorDict | torch.Tensor):
        self._triangulation[key] = value

    def batch_size(self):
        """return batch_size of triangulation tensordict"""
        return self._triangulation.batch_size

    def _triangle_to_tensordict(self, mesh_dict: dict[str, Any]):
        """Convert a mesh dictionary from 'triangle' library to a TensorDict"""
        key_map = {
            "vertices": ("vertices", "coordinates"),
            "vertex_markers": ("vertices", "markers"),
            "triangles": ("cells", "vertices"),
            "neighbors": ("cells", "neighbors"),
            "edges": ("edges", "vertices"),
            "edge_markers": ("edges", "markers"),
        }

        sub_dictionaries = {
            "vertices": {},
            "cells": {},
            "edges": {},
        }

        for key, value in mesh_dict.items():
            if key in key_map:
                subname, new_key = key_map[key]
                if value.dtype == int32:
                    sub_dictionaries[subname][new_key] = torch.tensor(
                        value, dtype=torch.int
                    )
                elif value.dtype == float:
                    sub_dictionaries[subname][new_key] = torch.tensor(
                        value, dtype=torch.get_default_dtype()
                    )
                elif isinstance(value, torch.Tensor):
                    sub_dictionaries[subname][new_key] = value

        mesh_tensordict = tensordict.TensorDict(
            {
                name: (
                    tensordict.TensorDict(
                        content, batch_size=[len(next(iter(content.values())))]
                    )
                    if content
                    else tensordict.TensorDict({})
                )
                for name, content in sub_dictionaries.items()
            },
            batch_size=[],
        )

        return mesh_tensordict

    def _build_optional_parameters(
        self, triangulation: tensordict.TensorDict
    ) -> tensordict.TensorDict:
        """Compute parameters that are not in mesh dict."""

        triangulation["cells", "coordinates"] = self.compute_coordinates_4_cells(
            triangulation["vertices", "coordinates"], triangulation["cells", "vertices"]
        )

        if "vertices" not in triangulation["edges"]:
            vertices_4_unique_edges, boundary_mask = self._compute_edges_vertices(
                triangulation,
            )
            triangulation["edges", "vertices"] = vertices_4_unique_edges
            triangulation["edges", "markers"] = boundary_mask

        interior_edges, boundary_edges = self._compute_interior_and_boundary_edges(
            triangulation
        )

        cells_length = self._compute_cells_min_length(triangulation)

        triangulation["interior_edges"] = interior_edges
        triangulation["boundary_edges"] = boundary_edges
        triangulation["cells", "length"] = cells_length
        triangulation["edges", "coordinates"] = self.compute_coordinates_4_cells(
            triangulation["vertices", "coordinates"], triangulation["edges", "vertices"]
        )

        return triangulation.auto_batch_size_()

    def _compute_interior_and_boundary_edges(
        self, triangulation: tensordict.TensorDict
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute interior and boundary edges."""

        vertices_4_boundary_edges, vertices_4_interior_edges = (
            self._compute_vertices_4_edges(triangulation)
        )

        cells_4_boundary_edges, cells_4_interior_edges = self._compute_cells_4_edges(
            triangulation, vertices_4_boundary_edges, vertices_4_interior_edges
        )

        coordinates_4_interior_edges = self.compute_coordinates_4_cells(
            triangulation["vertices", "coordinates"], vertices_4_interior_edges
        )

        coordinates_4_boundary_edges = self.compute_coordinates_4_cells(
            triangulation["vertices", "coordinates"], vertices_4_boundary_edges
        )

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
            triangulation["cells", "coordinates"], cells_4_interior_edges
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

        boundary_edges = tensordict.TensorDict(
            {
                "cells": cells_4_boundary_edges,
                "vertices": vertices_4_boundary_edges,
                "coordinates": coordinates_4_boundary_edges,
            }
        ).auto_batch_size_()
        interior_edges = tensordict.TensorDict(
            {
                "cells": cells_4_interior_edges,
                "vertices": vertices_4_interior_edges,
                "coordinates": coordinates_4_interior_edges,
                "length": interior_edges_length,
                "normals": normal_4_interior_edges,
            }
        ).auto_batch_size_()

        return interior_edges, boundary_edges

    def _compute_vertices_4_edges(
        self, triangulation: tensordict.TensorDict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute vertices for interior and boundary edges. boundary edges are identify for having
        a one in the marker for edges. every other marker is treated as interior."""

        vertices_4_edges = triangulation["edges", "vertices"]
        markers_4_edges = triangulation["edges", "markers"].squeeze(-1)

        vertices_4_boundary_edges = vertices_4_edges[markers_4_edges == 1]

        vertices_4_interior_edges = vertices_4_edges[markers_4_edges != 1]

        return vertices_4_boundary_edges, vertices_4_interior_edges

    def _compute_cells_4_edges(
        self,
        triangulation: tensordict.TensorDict,
        vertices_4_boundary_edges: torch.Tensor,
        vertices_4_interior_edges: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the cells for each edge. In the case of edges in the boundary of the domain,
        only 1 cells contain them, while 2 cells contains an edges for interior ones."""

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

    @staticmethod
    def compute_coordinates_4_cells(
        coordinates_4_vertices: torch.Tensor, vertices_4_cells: torch.Tensor
    ):
        """Compute the coordinates of the cells in the mesh."""
        return coordinates_4_vertices[vertices_4_cells]

    def _compute_edges_vertices(
        self, triangulation: tensordict.TensorDict
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute vertices for unique edges."""

        raise NotImplementedError("current implementation does not work as expected")

        vertices_4_edges = triangulation["cells", "vertices"][
            ..., self.edges_permutations
        ]

        edges_flat = vertices_4_edges.reshape(-1, 2)

        seen = set()
        unique_edges = []
        mask = torch.ones(edges_flat.shape[0])

        for i, e in enumerate(edges_flat):
            a, b = e.tolist()
            key = tuple(sorted((a, b)))
            if key not in seen:
                seen.add(key)
                unique_edges.append([a, b])
                mask[i] = 1

        unique_edges = torch.tensor(unique_edges)

        return unique_edges, mask

    def _compute_cells_min_length(
        self, triangulation: tensordict.TensorDict
    ) -> torch.Tensor:
        """For each cells, compute the smaller length of the edges."""
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

    @property
    @abc.abstractmethod
    def edges_permutations(self) -> torch.Tensor:
        """Return the local node vertices defining each edge of the element.
        the convection is the i-th node share numbering with the edge opposite to it."""
        raise NotImplementedError
