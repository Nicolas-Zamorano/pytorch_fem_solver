from typing import Tuple
import torch
from torch.nn.modules import Module

from torch_fem.problems.abstract_problem import AbstractProblem
from .deep_solver import DeepSolver
from torch_fem import Basis, ElementTri, ElementLine, InteriorEdgesBasis
import matplotlib.pyplot as plt
from ..model.neural_network import FeedForwardNeuralNetwork as NeuralNetwork
from ..model.model import Model


class RVPINNsSolver(DeepSolver):
    """Finite Element Method (FEM) solver."""

    def precompute_values(self) -> dict:
        """Precompute values needed for the FEM solver."""

        self.elements = ElementTri(self.p_order, self.q_order)
        self.basis = Basis(self.mesh, self.elements)

        exact_value = self.problem.exact(self.basis.integration_points)
        rhs_values = self.problem.rhs(self.basis.integration_points)
        exact_dx_value = self.problem.exact_dx(self.basis.integration_points)
        exact_dy_value = self.problem.exact_dy(self.basis.integration_points)

        if self._posteriori_error:
            self.edges_elements = ElementLine(self.p_order, self.q_order)
            self.edges_basis = InteriorEdgesBasis(self.mesh, self.edges_elements)

            self.jump_integration_points = (
                self.edges_basis.compute_jump_integration_points(delta=1e-6)
            )
            self.normals = self.basis.mesh["interior_edges", "normals"].unsqueeze(-2)
            self.element_size = self.basis.mesh["cells", "length"]
            self.edges_size = self.edges_basis.mesh["interior_edges", "length"]

        exact_l2_norm = torch.sqrt(
            torch.sum(
                self.basis.integrate_functional(
                    self.problem.precomputed_L2_norm,
                    exact_value,
                )
            )
        )

        exact_h1_norm = torch.sqrt(
            torch.sum(
                self.basis.integrate_functional(
                    self.problem.precomputed_H1_norm,
                    exact_value,
                    exact_dx_value,
                    exact_dy_value,
                )
            )
        )

        gram_matrix_inverse = torch.linalg.inv(
            self.basis.reduce(
                self.basis.integrate_bilinear_form(self.problem.bilinear_form)
            )
        )

        precomputed = {
            "exact_value": exact_value,
            "rhs_values": rhs_values,
            "exact_dx_value": exact_dx_value,
            "exact_dy_value": exact_dy_value,
            "exact_h1_norm": exact_h1_norm,
            "exact_L2_norm": exact_l2_norm,
            "gram_matrix_inverse": gram_matrix_inverse,
        }
        return precomputed

    def _training_step(self, neural_network: NeuralNetwork):

        if self._posteriori_error:
            neural_network_value, neural_network_grad, neural_network_laplacian = (
                neural_network.value_and_laplacian(self.basis.integration_points)
            )
            _, neural_network_grad_edges = neural_network.value_and_gradient(
                self.jump_integration_points
            )

            residual_vector = self.basis.reduce(
                self.basis.integrate_linear_form(
                    self.problem.residual,
                    gradient=neural_network_grad,
                    rhs_values=self.precomputed_values["rhs_values"],
                )
            )

            bulk_estimator = self.element_size**2 * self.basis.integrate_functional(
                self.problem.bulk_residual,
                laplacian=neural_network_laplacian,
                rhs_values=self.precomputed_values["rhs_values"],
            )

            jump_estimator = self.edges_size * self.edges_basis.integrate_functional(
                self.problem.jump_residual,
                gradient_for_jump=neural_network_grad_edges,
                normals_4_elements=self.normals,
            )

            loss_value = (
                residual_vector.T
                @ self.precomputed_values["gram_matrix_inverse"]
                @ residual_vector
                + torch.sum(bulk_estimator)
                + torch.sum(jump_estimator)
            )

        else:
            neural_network_value, neural_network_grad = (
                neural_network.value_and_gradient(self.basis.integration_points)
            )

            residual_vector = self.basis.reduce(
                self.basis.integrate_linear_form(
                    self.problem.residual,
                    gradient=neural_network_grad,
                    rhs_values=self.precomputed_values["rhs_values"],
                )
            )
            loss_value = (
                residual_vector.T
                @ self.precomputed_values["gram_matrix_inverse"]
                @ residual_vector
            )

        neural_network_dx, neural_network_dy = torch.split(neural_network_grad, 1, -1)

        h1_error = torch.sqrt(
            torch.sum(
                self.basis.integrate_functional(
                    self.problem.precomputed_H1_norm,
                    neural_network_value - self.precomputed_values["exact_value"],
                    neural_network_dx - self.precomputed_values["exact_dx_value"],
                    neural_network_dy - self.precomputed_values["exact_dy_value"],
                )
            )
        )

        relative_loss = (
            torch.sqrt(loss_value) / self.precomputed_values["exact_h1_norm"]
        )

        return (
            loss_value,
            relative_loss,
            h1_error / self.precomputed_values["exact_h1_norm"],
        )

    def compute_error(self, neural_network) -> Tuple[torch.Tensor, torch.Tensor]:

        neural_network_value, neural_network_grad = neural_network.value_and_gradient(
            self.basis.integration_points
        )

        neural_network_dx, neural_network_dy = torch.split(neural_network_grad, 1, -1)

        L2_error = self.basis.integrate_functional(
            self.problem.precomputed_L2_norm,
            neural_network_value - self.precomputed_values["exact_value"],  # type: ignore
        )

        H1_error = self.basis.integrate_functional(
            self.problem.precomputed_H1_norm,
            neural_network_value - self.precomputed_values["exact_value"],  # type: ignore
            neural_network_dx - self.precomputed_values["exact_dx_value"],
            neural_network_dy - self.precomputed_values["exact_dy_value"],
        )

        return L2_error, H1_error
