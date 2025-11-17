import os
from typing import Tuple
import torch
from .abstract_solver import AbstractSolver
from torch_fem import Basis, ElementTri
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection


class FemSolver(AbstractSolver):
    """Finite Element Method (FEM) solver."""

    def precompute_values(self) -> dict:
        """Precompute values needed for the FEM solver."""

        self.elements = ElementTri(self.p_order, self.q_order)
        self.basis = Basis(self.mesh, self.elements)

        exact_value = self.problem.exact(self.basis.integration_points)
        rhs_values = self.problem.rhs(self.basis.integration_points)
        exact_dx_value = self.problem.exact_dx(self.basis.integration_points)
        exact_dy_value = self.problem.exact_dy(self.basis.integration_points)

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

        solution_tensor = self.basis.evalute_at_boundary(
            self.problem.dirichlet_boundary
        )

        precomputed = {
            "exact_value": exact_value,
            "rhs_values": rhs_values,
            "solution_tensor": solution_tensor,
            "exact_dx_value": exact_dx_value,
            "exact_dy_value": exact_dy_value,
            "exact_H1_norm": exact_h1_norm,
            "exact_L2_norm": exact_l2_norm,
        }
        return precomputed

    def solve(self) -> torch.Tensor:
        stiffness_matrix = self.basis.integrate_bilinear_form(
            self.problem.bilinear_form
        )
        load_vector = self.basis.integrate_linear_form(
            self.problem.linear_form, rhs_values=self.precomputed_values["rhs_values"]
        )
        solution = self.basis.solve(
            stiffness_matrix,
            load_vector,
            solution=self.precomputed_values["solution_tensor"],
        )
        return solution

    def compute_error(self, numerical_solution) -> Tuple[torch.Tensor, torch.Tensor]:
        interpolation_solution, interpolation_solution_grad = self.basis.interpolate(
            self.basis, numerical_solution
        )

        interpolation_solution_dx, interpolation_solution_dy = torch.split(
            interpolation_solution_grad, 1, -1  # type: ignore
        )

        L2_error = self.basis.integrate_functional(
            self.problem.precomputed_L2_norm,
            interpolation_solution - self.precomputed_values["exact_value"],  # type: ignore
        )

        H1_error = self.basis.integrate_functional(
            self.problem.precomputed_H1_norm,
            interpolation_solution - self.precomputed_values["exact_value"],  # type: ignore
            interpolation_solution_dx - self.precomputed_values["exact_dx_value"],
            interpolation_solution_dy - self.precomputed_values["exact_dy_value"],
        )

        return L2_error, H1_error

    def plot(self, numerical_solution: torch.Tensor):
        L2_error, H1_error = self.compute_error(numerical_solution)

        relative_L2_error = L2_error.sum().sqrt() / self.precomputed_values[
            "exact_L2_norm"
        ].squeeze(-1)
        relative_H1_error = H1_error.sum().sqrt() / self.precomputed_values[
            "exact_H1_norm"
        ].squeeze(-1)

        coordinates_4_triangles = self.mesh["cells", "coordinates"]
        coordinates_4_vertices = self.mesh["vertices", "coordinates"]
        exact_value = self.problem.exact(coordinates_4_vertices).squeeze(-1).numpy()

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
            numerical_solution.squeeze(-1).numpy(),
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

        figure_error, axes_error = plt.subplots(1, 2, figsize=(10, 4))

        l2_error_surface = PolyCollection(
            coordinates_4_triangles,
            array=L2_error.sqrt().squeeze(-1).numpy(),
            cmap="viridis",
            edgecolor="black",
            linewidths=0.2,
        )
        axes_error[0].add_collection(l2_error_surface)
        axes_error[0].set_xlim(x_min, x_max)
        axes_error[0].set_ylim(y_min, y_max)
        axes_error[0].set_aspect("equal")
        axes_error[0].set_title(
            r"Relative L2 error = {:.4e}".format(relative_L2_error.item())
        )
        figure_error.colorbar(l2_error_surface, ax=axes_error[0])

        h1_error_surface = PolyCollection(
            coordinates_4_triangles,
            array=H1_error.sqrt().squeeze(-1).numpy(),
            cmap="viridis",
            edgecolor="black",
            linewidths=0.2,
        )
        axes_error[1].add_collection(h1_error_surface)
        axes_error[1].set_xlim(x_min, x_max)
        axes_error[1].set_ylim(y_min, y_max)
        axes_error[1].set_aspect("equal")
        axes_error[1].set_title(
            r"Relative H1 error = {:.4e}".format(relative_H1_error.item())
        )
        figure_error.colorbar(h1_error_surface, ax=axes_error[1])

        figure_error.tight_layout()

        return figure_solution, figure_error
