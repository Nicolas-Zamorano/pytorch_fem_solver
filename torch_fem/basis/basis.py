"""Class for standard basis representation"""

from typing import Optional, Callable, Tuple
import torch
from ..mesh.abstract_mesh import AbstractMesh
from ..element.abstract_element import AbstractElement
from .interior_edges_basis import InteriorEdgesBasis
from .abstract_basis import AbstractBasis


class Basis(AbstractBasis):
    """Class for standard basis representation"""

    def _compute_dofs(
        self,
        mesh: AbstractMesh,
        element: AbstractElement,
    ):

        coordinates_4_vertices = mesh["vertices", "coordinates"]
        vertices_4_cells = mesh["cells", "vertices"]
        markers_4_vertices = mesh["vertices", "markers"]

        if element.polynomial_order == 1:

            coords_4_global_dofs = coordinates_4_vertices
            global_dofs_4_elements = vertices_4_cells
            nodes_4_boundary_dofs = markers_4_vertices

        elif element.polynomial_order == 2:

            vertices_4_edges = mesh["edges", "vertices"]

            coordinates_4_new_dofs = mesh["edges", "coordinates"].mean(-2)

            new_dofs_enumeration = (
                torch.arange(vertices_4_edges.shape[0])
                + coordinates_4_vertices.shape[-2]
            )

            vertices_4_non_unique_edges = vertices_4_cells[..., mesh.edges_permutations]

            vertices_4_non_unique_edges_sorted, _ = vertices_4_non_unique_edges.sort(
                dim=-1
            )
            vertices_4_edges_sorted, _ = vertices_4_edges.sort(dim=-1)

            vertices_offset = vertices_4_cells.max() + 1
            vertices_keys = (
                vertices_4_non_unique_edges_sorted[..., 0] * vertices_offset
                + vertices_4_non_unique_edges_sorted[..., 1]
            )
            edge_keys = (
                vertices_4_edges_sorted[:, 0] * vertices_offset
                + vertices_4_edges_sorted[:, 1]
            )

            map_dict = -torch.ones(vertices_offset * vertices_offset, dtype=torch.int64)

            map_dict[edge_keys] = torch.arange(
                vertices_4_edges.shape[0],
                dtype=torch.int64,
            )

            global_edge_ids = map_dict[vertices_keys]

            vertices_4_new_dofs = new_dofs_enumeration[global_edge_ids]

            first_vertices_4_edges, second_vertices_4_edges = torch.unbind(
                vertices_4_edges, dim=-1
            )

            vertices_4_first_edge = torch.stack(
                [first_vertices_4_edges, new_dofs_enumeration], dim=-1
            )
            vertices_4_second_edge = torch.stack(
                [new_dofs_enumeration, second_vertices_4_edges], dim=-1
            )

            self.vertices_4_new_edges = torch.cat(
                [vertices_4_first_edge, vertices_4_second_edge], dim=-2
            )

            new_markers_4_new_dofs = mesh["edges", "markers"]

            coords_4_global_dofs = torch.cat(
                [coordinates_4_vertices, coordinates_4_new_dofs], dim=-2
            )
            global_dofs_4_elements = torch.cat(
                [vertices_4_cells, vertices_4_new_dofs], dim=-1
            )
            nodes_4_boundary_dofs = torch.cat(
                [markers_4_vertices, new_markers_4_new_dofs], dim=-2
            )

        elif element.polynomial_order == 3:

            vertices_4_edges = mesh["edges", "vertices"]

            (
                coordinates_4_vertices_first_vertex,
                coordinates_4_vertices_second_vertex,
            ) = torch.unbind(mesh["edges", "coordinates"], dim=-2)

            coordinates_4_new_edge_dofs = torch.stack(
                [
                    coordinates_4_vertices_first_vertex * 2 / 3
                    + coordinates_4_vertices_second_vertex * 1 / 3,
                    coordinates_4_vertices_first_vertex * 1 / 3
                    + coordinates_4_vertices_second_vertex * 2 / 3,
                ],
                dim=-2,
            ).reshape(-1, 2)

            # Cell-center DOFs (one per cell)
            coordinates_4_cell_center_dofs = mesh["cells", "coordinates"].mean(dim=-2)

            coordinates_4_new_dofs = torch.cat(
                [coordinates_4_new_edge_dofs, coordinates_4_cell_center_dofs],
                dim=-2,
            )

            # Enumerate edge DOFs
            new_edge_dofs_enumeration = (
                torch.arange(vertices_4_edges.shape[0] * 2)
                + coordinates_4_vertices.shape[-2]
            )

            # Enumerate cell-center DOFs
            new_cell_dofs_enumeration = (
                torch.arange(vertices_4_cells.shape[0])
                + coordinates_4_vertices.shape[-2]
                + vertices_4_edges.shape[0] * 2
            )

            vertices_4_non_unique_edges = vertices_4_cells[..., mesh.edges_permutations]

            first_vertices_4_edges, second_vertices_4_edges = torch.unbind(
                vertices_4_edges, dim=-1
            )

            vertices_4_second_edge = new_edge_dofs_enumeration.reshape(-1, 2)

            first_vertices_4_second_edge, second_vertices_4_second_edge = torch.unbind(
                vertices_4_second_edge, dim=-1
            )

            vertices_4_first_edge = torch.stack(
                [first_vertices_4_edges, first_vertices_4_second_edge], dim=-1
            )
            vertices_4_third_edge = torch.stack(
                [second_vertices_4_second_edge, second_vertices_4_edges], dim=-1
            )

            self.vertices_4_new_edges = torch.cat(
                [vertices_4_first_edge, vertices_4_second_edge, vertices_4_third_edge],
                dim=-2,
            )

            vertices_4_non_unique_edges_sorted, _ = vertices_4_non_unique_edges.sort(
                dim=-1
            )

            vertices_4_edges_sorted, _ = vertices_4_edges.sort(dim=-1)

            vertices_offset = vertices_4_cells.max() + 1
            vertices_keys = (
                vertices_4_non_unique_edges_sorted[..., 0] * vertices_offset
                + vertices_4_non_unique_edges_sorted[..., 1]
            )
            edge_keys = (
                vertices_4_edges_sorted[:, 0] * vertices_offset
                + vertices_4_edges_sorted[:, 1]
            )

            map_dict = -torch.ones(vertices_offset * vertices_offset, dtype=torch.int64)

            map_dict[edge_keys] = torch.arange(
                vertices_4_edges.shape[0],
                dtype=torch.int64,
            )

            global_edge_ids = map_dict[vertices_keys]

            # For polynomial order 3, we have 2 DOFs per edge
            # Create indices for both DOFs: [edge_id*2, edge_id*2+1]
            global_edge_ids_expanded = torch.stack(
                [global_edge_ids * 2, global_edge_ids * 2 + 1], dim=-1
            )
            vertices_4_new_edge_dofs = new_edge_dofs_enumeration[
                global_edge_ids_expanded.reshape(*global_edge_ids.shape[:-1], -1)
            ]

            # Cell-center DOF indices (one per cell)
            vertices_4_cell_center_dofs = new_cell_dofs_enumeration.unsqueeze(-1)

            # Combine edge DOFs and cell-center DOFs
            vertices_4_new_dofs = torch.cat(
                [vertices_4_new_edge_dofs, vertices_4_cell_center_dofs], dim=-1
            )

            # Markers: edge DOFs inherit edge markers, cell DOFs are interior (marker=0)
            new_markers_4_edge_dofs = mesh["edges", "markers"].repeat_interleave(
                2, dim=-2
            )
            new_markers_4_cell_dofs = torch.zeros(
                (vertices_4_cells.shape[0], 1), dtype=mesh["edges", "markers"].dtype
            )
            new_markers_4_new_dofs = torch.cat(
                [new_markers_4_edge_dofs, new_markers_4_cell_dofs], dim=-2
            )

            coords_4_global_dofs = torch.cat(
                [coordinates_4_vertices, coordinates_4_new_dofs], dim=-2
            )
            global_dofs_4_elements = torch.cat(
                [vertices_4_cells, vertices_4_new_dofs], dim=-1
            )
            nodes_4_boundary_dofs = torch.cat(
                [markers_4_vertices, new_markers_4_new_dofs], dim=-2
            )

        else:
            raise NotImplementedError("Polynomial order not implemented")

        coords_4_elements = mesh.compute_coordinates_4_cells(
            coords_4_global_dofs, global_dofs_4_elements
        )

        return (
            coords_4_global_dofs,
            global_dofs_4_elements,
            nodes_4_boundary_dofs,
            coords_4_elements,
        )

    def _compute_basis_parameters(
        self, coords4global_dofs, global_dofs4elements, nodes4boundary_dofs
    ):

        nb_global_dofs = coords4global_dofs.size(-2)
        nb_local_dofs = global_dofs4elements.size(-1)

        inner_dofs = torch.nonzero(nodes4boundary_dofs != 1, as_tuple=True)[-2]
        boundary_dofs = torch.nonzero(nodes4boundary_dofs == 1, as_tuple=True)[-2]

        rows_idx = global_dofs4elements.repeat(1, 1, nb_local_dofs).reshape(-1)
        cols_idx = global_dofs4elements.repeat_interleave(nb_local_dofs).reshape(-1)

        form_idx = global_dofs4elements.reshape(-1)

        return {
            "bilinear_form_shape": (nb_global_dofs, nb_global_dofs),
            "bilinear_form_idx": (rows_idx, cols_idx),
            "linear_form_shape": (nb_global_dofs, 1),
            "linear_form_idx": (form_idx,),
            "inner_dofs": inner_dofs,
            "boundary_dofs": boundary_dofs,
            "nb_dofs": nb_global_dofs,
        }

    def _compute_jacobian_map(self, mesh, element):
        return mesh["cells", "coordinates"].mT @ element.barycentric_map_gradient

    def _compute_integration_points(self, mesh, bar_coords):
        return bar_coords.mT @ mesh["cells", "coordinates"].unsqueeze(-3)

    def _compute_integral_weights(self, element, det_map_jacobian):
        return (
            element.reference_element_area * element.gaussian_weights * det_map_jacobian
        )

    def interpolate(
        self, basis: AbstractBasis, tensor: Optional[torch.Tensor] = None
    ) -> (
        Tuple[torch.Tensor, torch.Tensor]
        | Tuple[
            Callable[..., torch.Tensor],
            Callable[..., torch.Tensor],
        ]
    ):
        """Interpolate a tensor from the current basis to another basis."""
        if basis is self:
            indices_4_dofs = self.global_dofs4elements.unsqueeze(-2)

            v = self.v
            v_grad = self.v_grad

        elif basis.__class__ == Basis and basis is not self:

            elements_mask = basis.mesh["cells", "markers"].squeeze(-1).type(torch.int)

            # coords4elements_first_node = self.coords4elements[..., [0], :][
            #     elements_mask
            # ].unsqueeze(-3)

            # inv_map_jacobian = self._inv_map_jacobian[elements_mask]

            # # For computing the inverse mapping of the integrations points of the interior edges,
            # # is necessary that tensor are in the size (N_E, 2, q_E, N_f, N_d)
            # # (2 meaning the triangle that share and edge).

            # new_integrations_points = self._element.compute_inverse_map(
            #     coords4elements_first_node, basis.integration_points, inv_map_jacobian
            # )

            # new_bar_coords = self._element.compute_barycentric_coordinates(
            #     new_integrations_points
            # ).squeeze(-3)

            # v, v_grad = self._element.compute_shape_functions(
            #     new_bar_coords, inv_map_jacobian
            # )

            v = self.v[elements_mask].unsqueeze(-3)
            v_grad = self.v_grad[elements_mask]

            indices_4_dofs = self.global_dofs4elements[elements_mask].unsqueeze(-2)

        elif basis.__class__ == InteriorEdgesBasis:

            cells_4_interior_edges = basis.mesh["interior_edges", "cells"]

            coordinates_4_cells_first_vertex = basis.mesh.compute_coordinates_4_cells(
                self.mesh["cells", "coordinates"][..., [0], :],
                cells_4_interior_edges,
            ).unsqueeze(-3)

            inv_map_jacobian = basis.mesh.compute_coordinates_4_cells(
                self._inv_map_jacobian, cells_4_interior_edges
            )

            integrations_points = basis.integration_points.unsqueeze(-4)

            # For computing the inverse mapping of the integrations points of the interior edges,
            # is necessary that tensor are in the size (N_E, 2, q_E, N_f, N_d)
            # (2 meaning the triangle that share and edge).

            new_integrations_points = self._element.compute_inverse_map(
                coordinates_4_cells_first_vertex,
                integrations_points,
                inv_map_jacobian,
            )

            new_bar_coords = self._element.compute_barycentric_coordinates(
                new_integrations_points
            ).squeeze(-3)

            new_v, new_v_grad = self._element.compute_shape_functions(
                new_bar_coords, inv_map_jacobian
            )

            v = new_v
            v_grad = new_v_grad

            indices_4_dofs = basis.mesh.compute_coordinates_4_cells(
                basis.mesh["cells", "vertices"], cells_4_interior_edges
            ).unsqueeze(-2)

        else:
            raise NotImplementedError("Interpolation for this basis not implemented")

        if tensor is not None:

            interpolation = (tensor[indices_4_dofs] * v).sum(-2, keepdim=True)

            interpolation_grad = (tensor[indices_4_dofs] * v_grad).sum(-2, keepdim=True)

            return interpolation, interpolation_grad

        coordinates_4_dofs = self.coords4global_dofs

        def interpolator(
            function: Callable[[torch.Tensor], torch.Tensor],
        ) -> torch.Tensor:
            return (function(coordinates_4_dofs)[indices_4_dofs] * v).sum(
                -2, keepdim=True
            )

        def interpolator_grad(
            function: Callable[[torch.Tensor], torch.Tensor],
        ) -> torch.Tensor:
            return (function(coordinates_4_dofs)[indices_4_dofs] * v_grad).sum(
                -2, keepdim=True
            )

        return interpolator, interpolator_grad
