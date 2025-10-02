"""Abstract class for basis representation"""

import abc
from typing import Callable, Tuple, Any, Optional
import torch
from ..mesh.abstract_mesh import AbstractMesh
from ..element.abstract_element import AbstractElement


class AbstractBasis(abc.ABC):
    """Abstract class for basis representation"""

    def __init__(self, mesh: AbstractMesh, element: AbstractElement):

        self._element = element
        self.mesh = mesh

        (
            self.v,
            self.v_grad,
            self.integration_points,
            self._dx,
            self._inv_map_jacobian,
        ) = self._compute_integral_values(mesh, element)

        (
            self._coords4global_dofs,
            self._global_dofs4elements,
            self._nodes4boundary_dofs,
            self._coords4elements,
        ) = self._compute_dofs(
            mesh,
            element,
        )

        self._basis_parameters = self._compute_basis_parameters(
            self._coords4global_dofs,
            self._global_dofs4elements,
            self._nodes4boundary_dofs,
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

        _dx = self._compute_integral_weights(element, det_map_jacobian)

        return v, v_grad, integration_points, _dx, inv_map_jacobian

    def integrate_functional(
        self,
        function: Callable[..., torch.Tensor],
        *args: Optional[Any],
        **kwargs: Optional[Any],
    ) -> torch.Tensor:
        """Integrate a given functional over the mesh elements"""
        return (function(self, *args, **kwargs) * self._dx).sum(-3).sum(-2)

    def integrate_bilinear_form(
        self,
        function: Callable[..., torch.Tensor],
        *args: Optional[Any],
        **kwargs: Optional[Any],
    ) -> torch.Tensor:
        """Integrate a given bilinear form over the mesh elements"""
        global_matrix = torch.zeros(self._basis_parameters["bilinear_form_shape"])

        local_matrix = (function(self, *args, **kwargs) * self._dx).sum(-3)

        xd = self._basis_parameters["bilinear_form_idx"]

        global_matrix.index_put_(
            self._basis_parameters["bilinear_form_idx"],
            self.reshape_for_assembly(local_matrix, "bilinear"),
            accumulate=True,
        )

        return global_matrix

    def integrate_linear_form(
        self,
        function: Callable[..., torch.Tensor],
        *args: Optional[Any],
        **kwargs: Optional[Any],
    ) -> torch.Tensor:
        """Integrate a given linear form over the mesh elements"""
        integral_value = torch.zeros(self._basis_parameters["linear_form_shape"])

        integrand_value = (function(self, *args, **kwargs) * self._dx).sum(-3)

        integral_value.index_put_(
            self._basis_parameters["linear_form_idx"],
            self.reshape_for_assembly(integrand_value, "linear"),
            accumulate=True,
        )

        return integral_value

    def reduce(self, tensor: torch.Tensor):
        """Reduce a tensor to only include inner degrees of freedom"""
        idx = self._basis_parameters["inner_dofs"]
        return tensor[idx, :][:, idx] if tensor.size(-1) != 1 else tensor[idx]

    @abc.abstractmethod
    def _compute_dofs(
        self,
        mesh: AbstractMesh,
        element: AbstractElement,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the degrees of freedom for the basis functions"""
        raise NotImplementedError

    @abc.abstractmethod
    def _compute_basis_parameters(
        self,
        coords4global_dofs: torch.Tensor,
        global_dofs4elements: torch.Tensor,
        nodes4boundary_dofs: torch.Tensor,
    ) -> dict:
        """Compute parameters related to the basis functions"""
        raise NotImplementedError

    @abc.abstractmethod
    def _compute_jacobian_map(
        self, mesh: AbstractMesh, element: AbstractElement
    ) -> torch.Tensor:
        """Compute the jacobian of the map that maps the local element to the physical one."""
        raise NotImplementedError

    @abc.abstractmethod
    def _compute_integration_points(
        self, mesh: AbstractMesh, bar_coords: torch.Tensor
    ) -> torch.Tensor:
        """Compute the integration points, applying the map to the local quadrature points
        to obtain the physical integration points for each element"""
        raise NotImplementedError

    @abc.abstractmethod
    def _compute_integral_weights(
        self, element: AbstractElement, det_map_jacobian: torch.Tensor
    ) -> torch.Tensor:
        """Compute the integration weight, composing of the quadrature weights, area of the
        reference element, the determent of the jacobian of the map, as well other quantities
        """
        raise NotImplementedError

    def reshape_for_assembly(
        self, local_matrices: torch.Tensor, form: str
    ) -> torch.Tensor:
        """reshape local matrices tensor to be compute for assembly"""
        if form == "bilinear":
            return local_matrices.reshape(-1)
        elif form == "linear":
            return local_matrices.reshape(-1, 1)
        else:
            raise NotImplementedError(f"Unknown form type: {format(form)}")

    def solution_tensor(self) -> torch.Tensor:
        """return a empty vector with size (nb_dofs, 1)."""
        return torch.zeros(self._basis_parameters["linear_form_shape"])

    def solve(
        self,
        matrix: torch.Tensor,
        solution: torch.Tensor,
        vector: torch.Tensor,
        only_inner_dofs: bool = True,
    ) -> torch.Tensor:
        """Solve A*x=b, if only_inner_dofs used, solve reduced system."""
        if only_inner_dofs is True:
            matrix = self.reduce(matrix)
            vector = self.reduce(vector)

        solution[
            self._basis_parameters["inner_dofs"]
        ] += torch.linalg.solve(  # pylint: disable=not-callable
            matrix, vector
        )

        return solution
