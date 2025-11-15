import abc
import torch
from ..problems import AbstractProblem
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection


class AbstractSolver(abc.ABC):
    """Abstract base class for solvers."""

    def __init__(
        self,
        mesh,
        p_order: int,
        q_order: int,
        problem: AbstractProblem,
    ):
        self.mesh = mesh
        self.p_order = p_order
        self.q_order = q_order
        self.problem = problem
        self.precomputed_values = self.precompute_values()

    @abc.abstractmethod
    def precompute_values(self) -> dict:
        raise NotImplementedError

    @abc.abstractmethod
    def compute_error(self, numerical_solution: torch.Tensor):
        """Compute the error between numerical and exact solution."""
        raise NotImplementedError

    @abc.abstractmethod
    def solve(self):
        """compute the numerical solution."""
        raise NotImplementedError

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

        plt.show()
