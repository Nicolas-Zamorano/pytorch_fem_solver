"""Class for basis representation on interior edges"""

import torch
import tensordict
from ..mesh.abstract_mesh import AbstractMesh
from ..element.abstract_element import AbstractElement
from .abstract_basis import AbstractBasis


class InteriorEdgesBasis(AbstractBasis):
    """Class for basis representation on interior edges"""

    def _compute_dofs(
        self,
        mesh: AbstractMesh,
        element: AbstractElement,
    ):

        if element.polynomial_order == 1:
            ## WARNING !!!! THIS IS NOT CORRECT, NEED TO FIX
            coords_4_global_dofs = mesh["vertices", "coordinates"]
            global_dofs_4_elements = mesh["cells", "vertices"]
            nodes4boundary_dofs = mesh["vertices", "markers"]
        else:
            raise NotImplementedError("Polynomial order not implemented")

        coords4elements = mesh.compute_coordinates_4_cells(
            coords_4_global_dofs, global_dofs_4_elements
        )

        return (
            coords_4_global_dofs,
            global_dofs_4_elements,
            nodes4boundary_dofs,
            coords4elements,
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

        return tensordict.TensorDict(
            {
                "bilinear_form_shape": (nb_global_dofs, nb_global_dofs),
                "bilinear_form_idx": (rows_idx, cols_idx),
                "linear_form_shape": (nb_global_dofs, 1),
                "linear_form_idx": (form_idx,),
                "inner_dofs": inner_dofs,
                "nb_dofs": nb_global_dofs,
            }
        )

    def _compute_jacobian_map(self, mesh, element):
        return mesh["interior_edges", "coordinates"].mT @ element.barycentric_grad

    def _compute_integration_points(self, mesh, bar_coords):
        return bar_coords.mT @ mesh["interior_edges", "coordinates"].unsqueeze(-3)

    def _compute_integral_weights(self, element, det_map_jacobian):
        return (
            element.reference_element_area * element.gaussian_weights * det_map_jacobian
        )
