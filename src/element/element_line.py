"""Class for 1D line element representation"""

import torch
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
        return torch.concat([0.5 * (1.0 - x), 0.5 * (1.0 + x)], dim=-1)

    def _compute_gauss_values(self):

        if self.integration_order == 2:

            nodes = 1.0 / torch.sqrt(torch.tensor(3.0))

            gaussian_nodes = torch.tensor([[-nodes], [nodes]])

            gaussian_weights = torch.tensor([[[0.5]], [[0.5]]])

        elif self.integration_order == 3:

            nodes = torch.sqrt(torch.tensor(3 / 5))

            gaussian_nodes = torch.tensor([[0], [-nodes], [nodes]])

            gaussian_weights = torch.tensor([[[8 / 18]], [[5 / 18]], [[5 / 18]]])

        else:

            raise NotImplementedError("Integration order not implemented")

        return gaussian_nodes, gaussian_weights.unsqueeze(0)

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

        det_map_jacobian = torch.norm(map_jacobian, dim=-2, keepdim=True)

        inv_map_jacobian = 1.0 / det_map_jacobian

        return det_map_jacobian.unsqueeze(-1), inv_map_jacobian
