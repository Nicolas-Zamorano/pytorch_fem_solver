"""Class for standard basis representation"""

from typing import Optional
import torch
import tensordict
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

        # need to be refactored to account for deprecation of get_edges_idx function
        # elif element.polynomial_order == 2:

        #     new_coords4dofs = (
        #         mesh.coords4nodes[mesh.edges_parameters["nodes4unique_edges"]]
        #     ).mean(-2)
        #     new_nodes4dofs = (
        #         mesh.edges_parameters["edges_idx"].reshape(
        #             mesh.mesh_parameters["nb_simplex"], 3
        #         )
        #         + mesh.mesh_parameters["nb_nodes"]
        #     )
        #     new_nodes4boundary_dofs = (
        #         mesh.edges_parameters["nodes_idx4boundary_edges"]
        #         + mesh.mesh_parameters["nb_nodes"]
        #     )

        #     coords4global_dofs = torch.cat([mesh.coords4nodes, new_coords4dofs], dim=-2)
        #     global_dofs4elements = torch.cat(
        #         [mesh.nodes4elements, new_nodes4dofs], dim=-1
        #     )
        #     nodes4boundary_dofs = torch.cat(
        #         [mesh.nodes4boundary, new_nodes4boundary_dofs], dim=-1
        #     )
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

    def interpolate(self, basis: AbstractBasis, tensor: Optional[torch.Tensor] = None):
        """Interpolate a tensor from the current basis to another basis."""
        if basis is self:
            vertices_4_cells_4_interior_edges = self.global_dofs4elements.unsqueeze(-2)

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
                self.inv_map_jacobian, cells_4_interior_edges
            )

            integration_points = basis.integration_points.unsqueeze(-3)

            # For computing the inverse mapping of the integrations points of the interior edges,
            # is necessary that tensor are in the size (N_T, q_T, q_E, N_f, N_d).

            new_integrations_points = self.element.compute_inverse_map(
                coordinates_4_cells_first_vertex, integration_points, inv_map_jacobian
            )

            bar_coords = self.element.compute_barycentric_coordinates(
                new_integrations_points.squeeze(-3)
            )

            v, v_grad = self.element.compute_shape_functions(
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

        else:

            nodes = self.coords4global_dofs

            def interpolator(function):
                return (function(nodes)[vertices_4_cells_4_interior_edges] * v).sum(
                    -2, keepdim=True
                )

            def interpolator_grad(function):
                return (
                    function(nodes)[vertices_4_cells_4_interior_edges] * v_grad
                ).sum(-2, keepdim=True)

            return interpolator, interpolator_grad
