from typing import Tuple, Callable
import torch
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

from .deep_solver import DeepSolver
from torch_fem import Basis, ElementTri, ElementLine, InteriorEdgesBasis
from ..model.neural_network import FeedForwardNeuralNetwork as NeuralNetwork


class FEINNsSolver(DeepSolver):
    """Finite Element Method (FEM) solver."""

    def precompute_values(
        self, mesh, polynomial_order, integral_order, error_mesh=None
    ):
        """Precompute values needed for the FEM solver."""

        basis, precomputed_values, error_basis = self._precompute_values(
            mesh, polynomial_order, integral_order, error_mesh
        )

        interpolation_function, grad_interpolation_function = basis.interpolate(
            basis, tensor=None
        )

        exact_value_dofs = self.problem.exact(basis.coordinates_4_global_dofs)
        precomputed_values["exact_value_dofs"] = exact_value_dofs
        precomputed_values["boundary_dofs"] = basis.basis_parameters["boundary_dofs"]

        self.interpolation_function = interpolation_function
        self.grad_interpolation_function = grad_interpolation_function

        if self._posteriori_error:
            self.edges_elements = ElementLine(polynomial_order, integral_order)
            self.edges_basis = InteriorEdgesBasis(mesh, self.edges_elements)

            _, self.grad_interpolation_edges_function = basis.interpolate(
                self.edges_basis
            )
            self.normals = basis.mesh["interior_edges", "normals"].unsqueeze(-2)
            self.element_size = basis.mesh["cells", "length"]
            self.edges_size = self.edges_basis.mesh["interior_edges", "length"]

        return basis, precomputed_values, error_basis

    def _compute_loss(self, neural_network):
        if self._posteriori_error:
            neural_network_value_dofs = neural_network(
                self.basis.coordinates_4_global_dofs
            )

            neural_network_value_dofs[self.precomputed_values["boundary_dofs"]] = (
                self.precomputed_values["exact_value_dofs"][
                    self.precomputed_values["boundary_dofs"]
                ]
            )

            neural_network_interpolated = self.interpolation_function(
                neural_network_value_dofs
            )

            neural_network_grad_interpolated = self.grad_interpolation_function(
                neural_network_value_dofs
            )
            neural_network_laplacian_interpolated = (
                neural_network_value_dofs[
                    self.basis.global_dofs_4_local_dofs.unsqueeze(-2)
                ]
                * self.basis.v_lap
            ).sum(-2, keepdim=True)

            neural_network_grad_edges = self.grad_interpolation_edges_function(neural_network_value_dofs)  # type: ignore

            residual_vector = self.basis.reduce(
                self.basis.integrate_linear_form(
                    self.problem.residual,
                    gradient=neural_network_grad_interpolated,
                    rhs_values=self.precomputed_values["rhs_values"],
                )
            )

            bulk_estimator = self.element_size**2 * self.basis.integrate_functional(
                self.problem.bulk_residual,
                laplacian=neural_network_laplacian_interpolated,
                rhs_values=self.precomputed_values["rhs_values"],
            )

            jump_estimator = self.edges_size * self.edges_basis.integrate_functional(
                self.problem.jump_residual,
                gradient_for_jump=neural_network_grad_edges,
                normals_4_elements=self.normals,
            )

            loss_value = (
                torch.sum(residual_vector**2)
                + torch.sum(bulk_estimator)
                + torch.sum(jump_estimator)
            )

        else:
            neural_network_value_dofs = neural_network(
                self.basis.coordinates_4_global_dofs
            )

            neural_network_value_dofs[self.precomputed_values["boundary_dofs"]] = (
                self.precomputed_values["exact_value_dofs"][
                    self.precomputed_values["boundary_dofs"]
                ]
            )

            neural_network_interpolated = self.interpolation_function(
                neural_network_value_dofs
            )

            neural_network_grad_interpolated = self.grad_interpolation_function(
                neural_network_value_dofs
            )

            residual_vector = self.basis.reduce(
                self.basis.integrate_linear_form(
                    self.problem.residual,
                    gradient=neural_network_grad_interpolated,
                    rhs_value=self.precomputed_values["rhs_values"],
                )
            )

            loss_value = torch.sum(residual_vector**2)

        return loss_value

    def compute_error(self, numerical_solution) -> Tuple[torch.Tensor, torch.Tensor]:

        neural_network_value_dofs = self._neural_network(
            self.basis.coordinates_4_global_dofs
        )

        neural_network_value_dofs[self.precomputed_values["boundary_dofs"]] = (
            self.precomputed_values["exact_value_dofs"][
                self.precomputed_values["boundary_dofs"]
            ]
        )

        neural_network_interpolated = self.interpolation_function(
            neural_network_value_dofs
        )

        neural_network_grad_interpolated = self.grad_interpolation_function(
            neural_network_value_dofs
        )

        neural_network_interpolated_dx, neural_network_interpolated_dy = torch.split(
            neural_network_grad_interpolated, 1, -1
        )

        L2_error = self.basis.integrate_functional(
            self.problem.precomputed_L2_norm,
            neural_network_interpolated - self.precomputed_values["exact_value"],  # type: ignore
        )

        H1_error = self.basis.integrate_functional(
            self.problem.precomputed_H1_norm,
            neural_network_interpolated - self.precomputed_values["exact_value"],  # type: ignore
            neural_network_interpolated_dx - self.precomputed_values["exact_dx_value"],
            neural_network_interpolated_dy - self.precomputed_values["exact_dy_value"],
        )

        return L2_error, H1_error
