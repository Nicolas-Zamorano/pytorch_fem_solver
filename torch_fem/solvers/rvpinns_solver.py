from typing import Tuple
from matplotlib.figure import Figure
import torch
from torch.jit._script import ScriptModule
from .deep_solver import DeepSolver
from torch_fem import ElementLine, InteriorEdgesBasis
import matplotlib.pyplot as plt
from ..model.neural_network import FeedForwardNeuralNetwork as NeuralNetwork


class RVPINNsSolver(DeepSolver):
    """Finite Element Method (FEM) solver."""

    def precompute_values(
        self, mesh, polynomial_order, integral_order, error_mesh=None
    ):
        """Precompute values needed for the FEM solver."""
        basis, precomputed_values, error_basis = self._precompute_values(
            mesh, polynomial_order, integral_order, error_mesh
        )

        gram_matrix_inverse = torch.linalg.inv(
            basis.reduce(basis.integrate_bilinear_form(self.problem.bilinear_form))
        )

        gram_matrix_patches_inverse = torch.linalg.inv(
            torch.diagflat(
                torch.diag(
                    basis.reduce(
                        basis.integrate_bilinear_form(self.problem.bilinear_form)
                    )
                )
            )
        )
        precomputed_values["gram_matrix_inverse"] = gram_matrix_inverse
        precomputed_values["gram_matrix_patches_inverse"] = gram_matrix_patches_inverse

        if self._posteriori_error:
            self.edges_elements = ElementLine(polynomial_order, integral_order)
            self.edges_basis = InteriorEdgesBasis(mesh, self.edges_elements)

            self.jump_integration_points = (
                self.edges_basis.compute_jump_integration_points(delta=1e-6)
            )
            self.normals = basis.mesh["interior_edges", "normals"].unsqueeze(-2)
            self.element_size = basis.mesh["cells", "length"]
            self.edges_size = self.edges_basis.mesh["interior_edges", "length"]

        return basis, precomputed_values, error_basis

    def _compute_loss(self, neural_network):
        if self._posteriori_error:
            _, neural_network_grad, neural_network_laplacian = (
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
            _, neural_network_grad = neural_network.value_and_gradient(
                self.basis.integration_points
            )

            residual_vector = self.basis.reduce(
                self.basis.integrate_linear_form(
                    self.problem.residual,
                    gradient=neural_network_grad,
                    rhs_values=self.precomputed_values["rhs_values"],
                )
            )
            loss_value = (
                residual_vector.mT
                @ self.precomputed_values["gram_matrix_inverse"]
                @ residual_vector
            )

        return loss_value
