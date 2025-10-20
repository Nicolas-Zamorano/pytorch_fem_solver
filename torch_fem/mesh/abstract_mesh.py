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

        triangulation_tensordict["cells", "coordinates"] = (
            self.compute_coordinates_4_cells(
                triangulation_tensordict["vertices", "coordinates"],
                triangulation_tensordict["cells", "vertices"],
            )
        )

        triangulation_tensordict["cells", "length"] = self._compute_cells_max_length(
            triangulation_tensordict
        )

        self._triangulation = triangulation_tensordict

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
                if isinstance(value, torch.Tensor):
                    sub_dictionaries[subname][new_key] = value
                elif value.dtype == int32:
                    sub_dictionaries[subname][new_key] = torch.tensor(
                        value, dtype=torch.int
                    )
                elif value.dtype == float:
                    sub_dictionaries[subname][new_key] = torch.tensor(
                        value, dtype=torch.get_default_dtype()
                    )

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

    def compute_edges_values(self):
        """Compute edges values for the triangulation."""

        if "vertices" not in self._triangulation["edges"]:
            vertices_4_unique_edges, boundary_mask = self._compute_edges_vertices(
                self._triangulation,
            )
            self._triangulation["edges", "vertices"] = vertices_4_unique_edges
            self._triangulation["edges", "markers"] = boundary_mask

        interior_edges, boundary_edges = self._compute_interior_and_boundary_edges(
            self._triangulation
        )

        self._triangulation["interior_edges"] = interior_edges
        self._triangulation["boundary_edges"] = boundary_edges
        self._triangulation["edges", "coordinates"] = self.compute_coordinates_4_cells(
            self._triangulation["vertices", "coordinates"],
            self._triangulation["edges", "vertices"],
        )

    def _compute_interior_and_boundary_edges(
        self, triangulation: tensordict.TensorDict
    ) -> Tuple[tensordict.TensorDict, tensordict.TensorDict]:
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

        interior_edges_length, normal_4_interior_edges = (
            self._compute_interior_edge_lengths_and_normals(
                coordinates_4_interior_edges,
                cells_4_interior_edges,
                triangulation["cells", "coordinates"],
            )
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

    @staticmethod
    def compute_coordinates_4_cells(
        coordinates_4_vertices: torch.Tensor, vertices_4_cells: torch.Tensor
    ):
        """Compute the coordinates of the cells in the mesh."""
        return coordinates_4_vertices[vertices_4_cells]

    @property
    @abc.abstractmethod
    def edges_permutations(self) -> torch.Tensor:
        """Return the local node vertices defining each edge of the element.
        the convection is the i-th node share numbering with the edge opposite to it."""
        raise NotImplementedError

    @abc.abstractmethod
    def _compute_edges_vertices(
        self, triangulation: tensordict.TensorDict
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute unique edges and boundary markers for edges."""
        raise NotImplementedError

    @abc.abstractmethod
    def _compute_cells_max_length(
        self, triangulation: tensordict.TensorDict
    ) -> torch.Tensor:
        """Compute the maximum length of each cell in the mesh."""
        raise NotImplementedError

    @abc.abstractmethod
    def _compute_vertices_4_edges(
        self, triangulation: tensordict.TensorDict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute vertices for interior and boundary edges. boundary edges are identify for having
        a one in the marker for edges. every other marker is treated as interior."""
        raise NotImplementedError

    @abc.abstractmethod
    def _compute_cells_4_edges(
        self,
        triangulation: tensordict.TensorDict,
        vertices_4_boundary_edges: torch.Tensor,
        vertices_4_interior_edges: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the cells for each edge. In the case of edges in the boundary of the domain,
        only one cell is associated to the edge."""
        raise NotImplementedError

    @abc.abstractmethod
    def _compute_interior_edge_lengths_and_normals(
        self,
        coordinates_4_interior_edges: torch.Tensor,
        cells_4_interior_edges: torch.Tensor,
        coordinates_4_cells: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the length and normal vector for each edge."""
        raise NotImplementedError
