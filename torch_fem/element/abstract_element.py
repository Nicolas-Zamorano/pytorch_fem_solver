"""Abstract class for element representation"""

import abc
import torch


class AbstractElement(abc.ABC):
    """Abstract class for element representation"""

    def __init__(self, polynomial_order: int, integration_order: int):

        self.polynomial_order = polynomial_order
        self.integration_order = integration_order

        self.gaussian_nodes, self.gaussian_weights = self._compute_gauss_values()

    def compute_inverse_map(
        self,
        first_node: torch.Tensor,
        integration_points: torch.Tensor,
        inv_map_jacobian: torch.Tensor,
    ):
        """Compute the inverse map from physical coordinates to reference coordinates"""

        return (integration_points - first_node) @ inv_map_jacobian.mT

    @abc.abstractmethod
    def compute_shape_functions(
        self, bar_coords: torch.Tensor, inv_map_jacobian: torch.Tensor
    ):
        """Compute the shape functions and their gradients at given barycentric coordinates"""
        raise NotImplementedError

    @abc.abstractmethod
    def _compute_gauss_values(self):
        """Compute the Gaussian integration points and weights"""
        raise NotImplementedError

    @abc.abstractmethod
    def compute_barycentric_coordinates(self, x: torch.Tensor):
        """Compute the barycentric coordinates of given points x"""
        raise NotImplementedError

    @abc.abstractmethod
    def compute_det_and_inv_map(self, map_jacobian: torch.Tensor):
        """Compute the determinant and inverse of the mapping Jacobian"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def reference_element_area(self):
        """Return the area (or length) of the reference element"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def barycentric_grad(self):
        """Return the gradients of the barycentric coordinates in the reference element"""
        raise NotImplementedError
