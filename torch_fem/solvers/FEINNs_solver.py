from typing import Tuple
import torch
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

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
            "exact_H1_norm": exact_h1_norm,
            "interpolation_function": interpolation_function,
            "grad_interpolation_function": grad_interpolation_function,
            "exact_value_dofs": exact_value_dofs,
            "boundary_dofs": boundary_dofs,
        }

        return precomputed

    def _training_step(self, neural_network: NeuralNetwork):
        if self._posteriori_error:
            neural_network_value_dofs, neural_network_gradient_dofs = (
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
            ](neural_network_gradient_dofs)

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
            torch.sqrt(loss_value) / self.precomputed_values["exact_H1_norm"]
        )

        return (
            loss_value,
            relative_loss,
            h1_error / self.precomputed_values["exact_H1_norm"],
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

    def plot(self, neural_network: NeuralNetwork):
        L2_error, H1_error = self.compute_error(neural_network)

        neural_network_value_dofs = neural_network(self.basis.coords_4_global_dofs)

        neural_network_value_dofs[self.precomputed_values["boundary_dofs"]] = (
            self.precomputed_values["exact_value_dofs"][
                self.precomputed_values["boundary_dofs"]
            ]
        )

        relative_L2_error = L2_error.sum().sqrt() / self.precomputed_values[
            "exact_L2_norm"
        ].squeeze(-1)
        relative_H1_error = H1_error.sum().sqrt() / self.precomputed_values[
            "exact_H1_norm"
        ].squeeze(-1)

        coordinates_4_triangles = self.mesh["cells", "coordinates"]
        coordinates_4_vertices = self.mesh["vertices", "coordinates"]
        exact_value = (
            self.problem.exact(coordinates_4_vertices).squeeze(-1).numpy(force=True)
        )
        coordinates_4_vertices = self.mesh["vertices", "coordinates"].numpy(force=True)

        x_min, x_max = (
            coordinates_4_vertices[:, 0].min(),
            coordinates_4_vertices[:, 0].max(),
        )
        y_min, y_max = (
            coordinates_4_vertices[:, 1].min(),
            coordinates_4_vertices[:, 1].max(),
        )
        z_exact_min, z_exact_max = exact_value.min(), exact_value.max()

        figure_solution = plt.figure(figsize=(10, 4))

        axis_numerical_solution = figure_solution.add_subplot(1, 2, 1, projection="3d")

        axis_numerical_solution.plot_trisurf(
            coordinates_4_vertices[:, 0],
            coordinates_4_vertices[:, 1],
            neural_network_value_dofs.squeeze(-1).numpy(force=True),
            triangles=coordinates_4_triangles,
            cmap="viridis",
            edgecolor="black",
            linewidth=0.2,
        )

        axis_numerical_solution.set_xlim(x_min, x_max)
        axis_numerical_solution.set_ylim(y_min, y_max)
        axis_numerical_solution.set_zlim(z_exact_min, z_exact_max)
        axis_numerical_solution.set_title("Numerical Solution")
        axis_numerical_solution.set_xlabel("x")
        axis_numerical_solution.set_ylabel("y")
        axis_numerical_solution.set_zlabel("u(x,y)")

        axis_exact_solution = figure_solution.add_subplot(1, 2, 2, projection="3d")
        axis_exact_solution.plot_trisurf(
            coordinates_4_vertices[:, 0],
            coordinates_4_vertices[:, 1],
            exact_value,
            triangles=coordinates_4_triangles,
            cmap="viridis",
            edgecolor="black",
            linewidth=0.2,
        )

        axis_exact_solution.set_xlim(x_min, x_max)
        axis_exact_solution.set_ylim(y_min, y_max)
        axis_exact_solution.set_zlim(z_exact_min, z_exact_max)
        axis_exact_solution.set_title("Exact Solution")
        axis_exact_solution.set_xlabel("x")
        axis_exact_solution.set_ylabel("y")
        axis_exact_solution.set_zlabel("u(x,y)")

        figure_solution.tight_layout()

        figure_error, (axis_L2, axis_H1) = plt.subplots(1, 2, figsize=(10, 4))

        l2_error_surface = PolyCollection(
            coordinates_4_triangles,
            array=L2_error.sqrt().squeeze(-1).numpy(force=True),
            cmap="viridis",
            edgecolor="black",
            linewidths=0.2,
        )
        axis_L2.add_collection(l2_error_surface)
        axis_L2.set_xlim(x_min, x_max)
        axis_L2.set_ylim(y_min, y_max)
        axis_L2.set_aspect("equal")
        axis_L2.set_title(r"L2 Error = {:.4e}".format(L2_error.sum().sqrt().item()))
        figure_error.colorbar(l2_error_surface, ax=axis_L2)

        h1_error_surface = PolyCollection(
            coordinates_4_triangles,
            array=H1_error.sqrt().squeeze(-1).numpy(force=True),
            cmap="viridis",
            edgecolor="black",
            linewidths=0.2,
        )
        axis_H1.add_collection(h1_error_surface)
        axis_H1.set_xlim(x_min, x_max)
        axis_H1.set_ylim(y_min, y_max)
        axis_H1.set_aspect("equal")
        axis_H1.set_title(r"H1 Error = {:.4e}".format(H1_error.sum().sqrt().item()))
        figure_error.colorbar(h1_error_surface, ax=axis_H1)

        figure_error.tight_layout()

        loss_history, validation_loss_history, accuracy_history = (
            self.get_training_history()
        )

        figure_training, (axis_history, axis_robustness) = plt.subplots(
            1, 2, figsize=(10, 4)
        )

        axis_history.plot(loss_history, "-", label=r"$\mathcal{L}(u_{\theta})$")
        axis_history.plot(
            validation_loss_history,
            "--",
            label=r"$\frac{\sqrt{\mathcal{L}(u_{\theta})}}{\|u\|_U}$",
        )
        axis_history.plot(
            accuracy_history,
            ":",
            label=r"$\frac{\|u-u_{\theta}\|_U}{\|u\|_U}$",
        )
        axis_history.legend()
        axis_history.set_yscale("log")
        axis_history.set_xlabel("Epochs")
        axis_history.set_ylabel("Value")
        axis_history.set_title("Training History")
        axis_history.grid(True)

        robustness = [
            valiation_loss / accuracy_history
            for valiation_loss, accuracy_history in zip(
                validation_loss_history, accuracy_history
            )
        ]

        axis_robustness.plot(
            robustness,
            ":",
            label=r"$\frac{\sqrt{\mathcal{L}(u_{\theta})}}{\|u-u_{\theta}\|_U}$",
        )
        axis_robustness.set_xlabel("Epochs")
        axis_robustness.set_ylabel("Value")
        axis_robustness.set_title("Training Robustness")
        axis_robustness.legend()
        axis_robustness.grid(True)
        figure_training.tight_layout()

        return (
            figure_solution,
            figure_error,
            figure_training,
        )
