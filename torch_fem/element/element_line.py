"""Class for 1D line element representation"""

import torch
from numpy.polynomial.legendre import leggauss
from .abstract_element import AbstractElement


class ElementLine(AbstractElement):
    """Class for 1D line element representation"""

    @property
    def barycentric_grad(self):
        return torch.tensor([[-0.5], [0.5]])

    @property
    def reference_element_area(self):
        return 2.0

    def compute_barycentric_coordinates(self, x: torch.Tensor):
        return torch.stack([0.5 * (1.0 - x), 0.5 * (1.0 + x)], dim=-2)

    def _compute_gauss_values(self):

        gaussian_nodes_np, gaussian_weights_np = leggauss(self.integration_order)

        gaussian_nodes = torch.Tensor(gaussian_nodes_np).reshape(-1, 1)

        # We use the convection of weights should sum to 1.
        gaussian_weights = torch.Tensor(0.5 * gaussian_weights_np).reshape(-1, 1, 1)

        return gaussian_nodes, gaussian_weights

    def compute_shape_functions(
        self, bar_coords: torch.Tensor, inv_map_jacobian: torch.Tensor
    ):

        if self.polynomial_order == 1:

            v = bar_coords

            v_grad = self.barycentric_grad @ inv_map_jacobian

        else:

            raise NotImplementedError("Polynomial order not implemented")

        return v, v_grad

    def compute_det_and_inv_map(self, map_jacobian: torch.Tensor):

        det_map_jacobian: torch.Tensor = (
            torch.linalg.vector_norm(  # pylint: disable=not-callable
                map_jacobian,
                dim=-2,
                keepdim=True,
            )
        )

        inv_map_jacobian = 1.0 / det_map_jacobian

        return det_map_jacobian.unsqueeze(-1), inv_map_jacobian
