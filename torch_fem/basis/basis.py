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

        if element.polynomial_order == 1:

            coords_4_global_dofs = mesh["vertices", "coordinates"]
            global_dofs_4_elements = mesh["cells", "vertices"]
            nodes_4_boundary_dofs = mesh["vertices", "markers"]

        elif element.polynomial_order == 2:

            coordinates_4_vertices = mesh["vertices", "coordinates"]

            coordinates_4_new_dofs = mesh["edges", "coordinates"].mean(-2)

            vertices_4_cells = mesh["cells", "vertices"]
            vertices_4_edges = mesh["edges", "vertices"]

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

            new_markers_4_new_dofs = mesh["edges", "markers"]

            coords_4_global_dofs = torch.cat(
                [mesh["vertices", "coordinates"], coordinates_4_new_dofs], dim=-2
            )
            global_dofs_4_elements = torch.cat(
                [mesh["cells", "vertices"], vertices_4_new_dofs], dim=-1
            )
            nodes_4_boundary_dofs = torch.cat(
                [mesh["vertices", "markers"], new_markers_4_new_dofs], dim=-2
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

        rows_idx = global_dofs4elements.repeat(1, 1, nb_local_dofs).reshape(-1)
        cols_idx = global_dofs4elements.repeat_interleave(nb_local_dofs).reshape(-1)

        form_idx = global_dofs4elements.reshape(-1)

        return {
            "bilinear_form_shape": (nb_global_dofs, nb_global_dofs),
            "bilinear_form_idx": (rows_idx, cols_idx),
            "linear_form_shape": (nb_global_dofs, 1),
            "linear_form_idx": (form_idx,),
            "inner_dofs": inner_dofs,
            "nb_dofs": nb_global_dofs,
        }

    def _compute_jacobian_map(self, mesh, element):
        return mesh["cells", "coordinates"].mT @ element.barycentric_grad

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
            Callable[[Callable[[torch.Tensor], torch.Tensor]], torch.Tensor],
            Callable[[Callable[[torch.Tensor], torch.Tensor]], torch.Tensor],
        ]
    ):
        """Interpolate a tensor from the current basis to another basis."""
        if basis is self:
            vertices_4_cells_4_interior_edges = self._global_dofs4elements.unsqueeze(-2)

            v = self.v
            v_grad = self.v_grad

        elif basis.__class__ == InteriorEdgesBasis:

            cells_4_interior_edges = basis.mesh["interior_edges", "cells"]

            vertices_4_cells_4_interior_edges = basis.mesh.compute_coordinates_4_cells(
                basis.mesh["cells", "vertices"], cells_4_interior_edges
            ).unsqueeze(-2)

            coordinates_4_cells_first_vertex = basis.mesh.compute_coordinates_4_cells(
                self.mesh["cells", "coordinates"][..., [0], :], cells_4_interior_edges
            ).unsqueeze(-3)

            inv_map_jacobian = basis.mesh.compute_coordinates_4_cells(
                self._inv_map_jacobian, cells_4_interior_edges
            )

            integration_points = basis.integration_points.unsqueeze(-3)

            # For computing the inverse mapping of the integrations points of the interior edges,
            # is necessary that tensor are in the size (N_T, q_T, q_E, N_f, N_d).

            new_integrations_points = self._element.compute_inverse_map(
                coordinates_4_cells_first_vertex, integration_points, inv_map_jacobian
            )

            bar_coords = self._element.compute_barycentric_coordinates(
                new_integrations_points.squeeze(-3)
            )

            v, v_grad = self._element.compute_shape_functions(
                bar_coords, inv_map_jacobian
            )
        else:
            raise NotImplementedError("Interpolation for this basis not implemented")

        if tensor is not None:

            interpolation = (tensor[vertices_4_cells_4_interior_edges] * v).sum(
                -2, keepdim=True
            )

            interpolation_grad = (
                tensor[vertices_4_cells_4_interior_edges] * v_grad
            ).sum(-2, keepdim=True)

            return interpolation, interpolation_grad

        nodes = self._coords4global_dofs

        def interpolator(
            function: Callable[[torch.Tensor], torch.Tensor],
        ) -> torch.Tensor:
            return (function(nodes)[vertices_4_cells_4_interior_edges] * v).sum(
                -2, keepdim=True
            )

        def interpolator_grad(
            function: Callable[[torch.Tensor], torch.Tensor],
        ) -> torch.Tensor:
            return (function(nodes)[vertices_4_cells_4_interior_edges] * v_grad).sum(
                -2, keepdim=True
            )

        return interpolator, interpolator_grad
