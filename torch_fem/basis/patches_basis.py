"""Basis class for Patches"""

import torch
from ..mesh.abstract_mesh import AbstractMesh
from ..element.abstract_element import AbstractElement
from .abstract_basis import AbstractBasis


class PatchesBasis(AbstractBasis):
    """Basis class for Patches"""

    def __init__(self, mesh, element):
        self.nb_patches = mesh.batch_size()[0]
        self.patches_idx = torch.arange(self.nb_patches).unsqueeze(-1)

        super().__init__(mesh, element)

    def _compute_dofs(
        self,
        mesh: AbstractMesh,
        element: AbstractElement,
    ):

        if element.polynomial_order == 1:

            coords_4_global_dofs = mesh["vertices", "coordinates"]
            global_dofs_4_elements = mesh["cells", "vertices"]
            nodes_4_boundary_dofs = mesh["vertices", "markers"]

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

        rows_idx = global_dofs4elements.repeat(1, 1, nb_local_dofs).reshape(
            self.nb_patches, -1
        )
        cols_idx = global_dofs4elements.repeat_interleave(
            nb_local_dofs, dim=-1
        ).reshape(self.nb_patches, -1)

        form_idx = global_dofs4elements.reshape(self.nb_patches, -1)

        return {
            "bilinear_form_shape": (
                self.nb_patches,
                nb_global_dofs,
                nb_global_dofs,
            ),
            "bilinear_form_idx": (self.patches_idx, rows_idx, cols_idx),
            "linear_form_shape": (self.nb_patches, nb_global_dofs, 1),
            "linear_form_idx": (
                self.patches_idx,
                form_idx,
            ),
            "inner_dofs": inner_dofs,
            "nb_dofs": nb_global_dofs,
        }

    def reshape_for_assembly(self, local_matrices: torch.Tensor, form: str):
        """reshape local matrices tensor to be compute for assembly"""
        if form == "bilinear":
            return local_matrices.reshape(self.nb_patches, -1)
        elif form == "linear":
            return local_matrices.reshape(self.nb_patches, -1, 1)
        else:
            raise NotImplementedError(f"Unknown form type: {format(form)}")

    def _compute_integral_weights(
        self, element: AbstractElement, det_map_jacobian: torch.Tensor
    ):
        return (
            element.reference_element_area * element.gaussian_weights * det_map_jacobian
        )

    def _compute_integration_points(self, mesh: AbstractMesh, bar_coords: torch.Tensor):
        return bar_coords.mT @ mesh["cells", "coordinates"].unsqueeze(-3)

    def _compute_jacobian_map(self, mesh, element):
        return mesh["cells", "coordinates"].mT @ element.barycentric_grad

    def reduce(self, tensor: torch.Tensor):
        idx = self.basis_parameters["inner_dofs"]
        return (
            tensor[self.patches_idx.squeeze(), idx, idx]
            if tensor.size(-1) != 1
            else tensor[self.patches_idx.squeeze(), idx]
        )
