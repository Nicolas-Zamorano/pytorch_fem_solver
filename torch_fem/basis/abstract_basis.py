"""Abstract class for basis representation"""

import abc
import torch
from ..mesh.abstract_mesh import AbstractMesh
from ..element.abstract_element import AbstractElement


class AbstractBasis(abc.ABC):
    """Abstract class for basis representation"""

    def __init__(self, mesh: AbstractMesh, element: AbstractElement):

        self.element = element
        self.mesh = mesh

        (
            self.v,
            self.v_grad,
            self.integration_points,
            self.dx,
            self.inv_map_jacobian,
        ) = self._compute_integral_values(mesh, element)

        (
            self.coords4global_dofs,
            self.global_dofs4elements,
            self.nodes4boundary_dofs,
            self.coords4elements,
        ) = self._compute_dofs(
            mesh,
            element,
        )

        self.basis_parameters = self._compute_basis_parameters(
            self.coords4global_dofs, self.global_dofs4elements, self.nodes4boundary_dofs
        )

    def _compute_integral_values(
        self,
        mesh: AbstractMesh,
        element: AbstractElement,
    ):
        """Compute the values for the numerical integration"""

        map_jacobian = self._compute_jacobian_map(mesh, element)

        det_map_jacobian, inv_map_jacobian = element.compute_det_and_inv_map(
            map_jacobian
        )

        bar_coords = element.compute_barycentric_coordinates(element.gaussian_nodes)

        v, v_grad = element.compute_shape_functions(bar_coords, inv_map_jacobian)

        integration_points = self._compute_integration_points(mesh, bar_coords)

        dx = self._compute_integral_weights(element, det_map_jacobian)

        return v, v_grad, integration_points, dx, inv_map_jacobian

    def integrate_functional(self, function, *args, **kwargs):
        """Integrate a given functional over the mesh elements"""
        return (function(self, *args, **kwargs) * self.dx).sum(-3).sum(-2)

    def integrate_bilinear_form(self, function, *args, **kwargs):
        """Integrate a given bilinear form over the mesh elements"""
        global_matrix = torch.zeros(self.basis_parameters["bilinear_form_shape"])

        local_matrix = (function(self, *args, **kwargs) * self.dx).sum(-3)

        global_matrix.index_put_(
            self.basis_parameters["bilinear_form_idx"],
            local_matrix.reshape(-1),
            accumulate=True,
        )

        return global_matrix

    def integrate_linear_form(self, function, *args, **kwargs):
        """Integrate a given linear form over the mesh elements"""
        integral_value = torch.zeros(self.basis_parameters["linear_form_shape"])

        integrand_value = (function(self, *args, **kwargs) * self.dx).sum(-3)

        integral_value.index_put_(
            self.basis_parameters["linear_form_idx"],
            integrand_value.reshape(-1, 1),
            accumulate=True,
        )

        return integral_value

    def reduce(self, tensor: torch.Tensor):
        """Reduce a tensor to only include inner degrees of freedom"""
        idx = self.basis_parameters["inner_dofs"]
        return tensor[idx, :][:, idx] if tensor.size(-1) != 1 else tensor[idx]

    @abc.abstractmethod
    def _compute_dofs(
        self,
        mesh: AbstractMesh,
        element: AbstractElement,
    ):
        """Compute the degrees of freedom for the basis functions"""
        raise NotImplementedError

    @abc.abstractmethod
    def _compute_basis_parameters(
        self, coords4global_dofs, global_dofs4elements, nodes4boundary_dofs
    ):
        """Compute parameters related to the basis functions"""
        raise NotImplementedError

    @abc.abstractmethod
    def _compute_jacobian_map(self, mesh: AbstractMesh, element: AbstractElement):
        """Compute the jacobian of the map that maps the local element to the physical one."""
        raise NotImplementedError

    @abc.abstractmethod
    def _compute_integration_points(self, mesh: AbstractMesh, bar_coords: torch.Tensor):
        """Compute the integration points, applying to map to the local quadrature points
        to obtain the physical integration points for each element"""
        raise NotImplementedError

    @abc.abstractmethod
    def _compute_integral_weights(
        self, element: AbstractMesh, det_map_jacobian: torch.Tensor
    ):
        """Compute the integration weight, composing of the quadrature weights, area of the
        reference element, the determent of the jacobian of the map, as well other quantities
        """
        raise NotImplementedError
