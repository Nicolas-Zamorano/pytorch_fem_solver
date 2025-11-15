from typing import Tuple
import torch

from .deep_solver import DeepSolver
from torch_fem import Basis, ElementTri, ElementLine, InteriorEdgesBasis
from ..model.neural_network import FeedForwardNeuralNetwork as NeuralNetwork


class FEINNsSolver(DeepSolver):
    """Finite Element Method (FEM) solver."""

    def precompute_values(self) -> dict:
        """Precompute values needed for the FEM solver."""

        self.elements = ElementTri(self.p_order, self.q_order)
        self.basis = Basis(self.mesh, self.elements)

        exact_value = self.problem.exact(self.basis.integration_points)
        exact_value_dofs = self.problem.exact(self.basis.coords_4_global_dofs)
        rhs_values = self.problem.rhs(self.basis.integration_points)
        exact_dx_value = self.problem.exact_dx(self.basis.integration_points)
        exact_dy_value = self.problem.exact_dy(self.basis.integration_points)

        if self._posteriori_error:
            self.edges_elements = ElementLine(self.p_order, self.q_order)
            self.edges_basis = InteriorEdgesBasis(self.mesh, self.edges_elements)

            _, self.grad_interpolation_edges_function = self.basis.interpolate(
                self.edges_basis
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

        interpolation_function, grad_interpolation_function = self.basis.interpolate(
            self.basis
        )

        boundary_dofs = self.basis.basis_parameters["boundary_dofs"].squeeze(-1)

        precomputed = {
            "exact_value": exact_value,
            "rhs_values": rhs_values,
            "exact_dx_value": exact_dx_value,
            "exact_dy_value": exact_dy_value,
            "exact_L2_norm": exact_l2_norm,
            "exact_h1_norm": exact_h1_norm,
            "interpolation_function": interpolation_function,
            "grad_interpolation_function": grad_interpolation_function,
            "exact_value_dofs": exact_value_dofs,
            "boundary_dofs": boundary_dofs,
        }

        return precomputed

    def _training_step(self, neural_network: NeuralNetwork):
        if self._posteriori_error:
            neural_network_value_dofs, neural_network_laplacian_dofs = (
                neural_network.value_and_gradient(self.basis.coords_4_global_dofs)
            )

            neural_network_value_dofs[self.precomputed_values["boundary_dofs"]] = (
                self.precomputed_values["exact_value_dofs"][
                    self.precomputed_values["boundary_dofs"]
                ]
            )

            neural_network_interpolated = self.precomputed_values[
                "interpolation_function"
            ](neural_network_value_dofs)

            neural_network_grad_interpolated = self.precomputed_values[
                "grad_interpolation_function"
            ](neural_network_value_dofs)

            neural_network_laplacian_interpolated = self.precomputed_values[
                "grad_interpolation_function"
            ](neural_network_laplacian_dofs)

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
            neural_network_value_dofs = neural_network(self.basis.coords_4_global_dofs)

            neural_network_value_dofs[self.precomputed_values["boundary_dofs"]] = (
                self.precomputed_values["exact_value_dofs"][
                    self.precomputed_values["boundary_dofs"]
                ]
            )

            neural_network_interpolated = self.precomputed_values[
                "interpolation_function"
            ](neural_network_value_dofs)

            neural_network_grad_interpolated = self.precomputed_values[
                "grad_interpolation_function"
            ](neural_network_value_dofs)

            residual_vector = self.basis.reduce(
                self.basis.integrate_linear_form(
                    self.problem.residual,
                    gradient=neural_network_grad_interpolated,
                    rhs_values=self.precomputed_values["rhs_values"],
                )
            )

            loss_value = torch.sum(residual_vector**2)

        neural_network_interpolated_dx, neural_network_interpolated_dy = torch.split(
            neural_network_grad_interpolated, 1, -1
        )

        h1_error = torch.sqrt(
            torch.sum(
                self.basis.integrate_functional(
                    self.problem.precomputed_H1_norm,
                    neural_network_interpolated
                    - self.precomputed_values["exact_value"],
                    neural_network_interpolated_dx
                    - self.precomputed_values["exact_dx_value"],
                    neural_network_interpolated_dy
                    - self.precomputed_values["exact_dy_value"],
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

        neural_network_value_dofs = neural_network(self.basis.coords_4_global_dofs)

        neural_network_value_dofs[self.precomputed_values["boundary_dofs"]] = (
            self.precomputed_values["exact_value_dofs"][
                self.precomputed_values["boundary_dofs"]
            ]
        )

        neural_network_interpolated = self.precomputed_values["interpolation_function"](
            neural_network_value_dofs
        )

        neural_network_grad_interpolated = self.precomputed_values[
            "grad_interpolation_function"
        ](neural_network_value_dofs)

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
