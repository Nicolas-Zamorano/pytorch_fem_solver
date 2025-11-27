import abc
from typing import Any, Optional, Tuple
import torch
from ..problems import AbstractProblem
from ..basis import AbstractBasis, Basis
from ..element import ElementTri
from ..mesh import AbstractMesh
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure


class AbstractSolver(abc.ABC):
    """Abstract base class for solvers."""

    def __init__(
        self,
        mesh: AbstractMesh,
        polynomial_order: int,
        integral_order: int,
        problem: AbstractProblem,
        error_mesh: Optional[AbstractMesh] = None,
    ):
        self.mesh = mesh
        self.polynomial_order = polynomial_order
        self.integral_order = integral_order
        self.problem = problem
        self.basis, self.precomputed_values, self.error_basis = self.precompute_values(
            mesh, polynomial_order, integral_order, error_mesh
        )

    def _precompute_values(
        self,
        mesh: AbstractMesh,
        polynomial_order: int,
        integral_order: int,
        error_mesh: Optional[AbstractMesh] = None,
    ) -> Tuple[AbstractBasis, dict[str, torch.Tensor], AbstractBasis]:
        """Precompute values needed for the FEM solver."""

        element = ElementTri(polynomial_order, integral_order)
        basis = Basis(mesh, element)

        if error_mesh is not None:
            error_element = ElementTri(polynomial_order, 2 * integral_order)
            error_basis = Basis(error_mesh, error_element)
        else:
            error_basis = basis

        exact_value = self.problem.exact(error_basis.integration_points)
        rhs_values = self.problem.rhs(basis.integration_points)
        exact_dx_value = self.problem.exact_dx(error_basis.integration_points)
        exact_dy_value = self.problem.exact_dy(error_basis.integration_points)

        exact_l2_norm = torch.sqrt(
            torch.sum(
                error_basis.integrate_functional(
                    self.problem.precomputed_L2_norm,
                    exact_value,
                )
            )
        )
        exact_h1_norm = torch.sqrt(
            torch.sum(
                error_basis.integrate_functional(
                    self.problem.precomputed_H1_norm,
                    exact_value,
                    exact_dx_value,
                    exact_dy_value,
                )
            )
        )

        precomputed_values = {
            "exact_value": exact_value,
            "rhs_values": rhs_values,
            "exact_dx_value": exact_dx_value,
            "exact_dy_value": exact_dy_value,
            "exact_H1_norm": exact_h1_norm,
            "exact_L2_norm": exact_l2_norm,
        }
        return basis, precomputed_values, error_basis

    def precompute_values(
        self,
        mesh: AbstractMesh,
        polynomial_order: int,
        integral_order: int,
        error_mesh: Optional[AbstractMesh] = None,
    ) -> Tuple[AbstractBasis, dict[str, torch.Tensor], AbstractBasis]:
        return self._precompute_values(
            mesh, polynomial_order, integral_order, error_mesh
        )

    def _plot(self, numerical_solution: torch.Tensor) -> Tuple[Figure, Figure]:
        """Plot the numerical solution over the mesh."""
        L2_error, H1_error = self.compute_error(numerical_solution)

        relative_L2_error = (
            L2_error.sum(-2, keepdim=True).sqrt()
            / self.precomputed_values["exact_L2_norm"]
        )
        relative_H1_error = (
            H1_error.sum(-2, keepdim=True).sqrt()
            / self.precomputed_values["exact_H1_norm"]
        )

        coordinates_4_vertices = self.error_basis.coordinates_4_global_dofs

        coordinates_4_triangles = self.error_basis.coordinates_4_local_dofs.numpy(
            force=True
        )

        exact_value = (
            self.problem.exact(coordinates_4_vertices).squeeze(-1).numpy(force=True)
        )

        x_min, x_max = (
            coordinates_4_vertices[:, 0].min(),
            coordinates_4_vertices[:, 0].max(),
        )
        y_min, y_max = (
            coordinates_4_vertices[:, 1].min(),
            coordinates_4_vertices[:, 1].max(),
        )

        figure_solution = plt.figure(figsize=(10, 4))

        axis_numerical_solution = figure_solution.add_subplot(1, 2, 1, projection="3d")

        axis_numerical_solution.plot_trisurf(
            coordinates_4_vertices[:, 0],
            coordinates_4_vertices[:, 1],
            numerical_solution.squeeze(-1).numpy(force=True),
            triangles=coordinates_4_triangles,
            edgecolor="black",
            linewidth=0.2,
        )

        axis_numerical_solution.set_xlim(x_min, x_max)
        axis_numerical_solution.set_ylim(y_min, y_max)
        axis_numerical_solution.set_title("Numerical Solution")
        axis_numerical_solution.set_xlabel(r"$x$")
        axis_numerical_solution.set_ylabel(r"$y$")
        axis_numerical_solution.set_zlabel(r"$u_\theta(x,y)$")

        axis_exact_solution = figure_solution.add_subplot(1, 2, 2, projection="3d")
        axis_exact_solution.plot_trisurf(
            coordinates_4_vertices[:, 0],
            coordinates_4_vertices[:, 1],
            exact_value,
            triangles=coordinates_4_triangles,
            edgecolor="black",
            linewidth=0.2,
        )

        axis_exact_solution.set_xlim(x_min, x_max)
        axis_exact_solution.set_ylim(y_min, y_max)
        axis_exact_solution.set_title("Exact Solution")
        axis_exact_solution.set_xlabel(r"$x$")
        axis_exact_solution.set_ylabel(r"$y$")
        axis_exact_solution.set_zlabel(r"$u(x,y)$")

        figure_solution.tight_layout()

        figure_error, (axis_L2_error, axis_H1_error) = plt.subplots(
            1, 2, figsize=(10, 4)
        )

        l2_error_surface = PolyCollection(
            coordinates_4_triangles[..., :3, :],  # type: ignore
            array=L2_error.sqrt().squeeze(-1).numpy(force=True),
            cmap="viridis",
            edgecolor="black",
            linewidths=0.2,
        )
        axis_L2_error.add_collection(l2_error_surface)
        axis_L2_error.set_xlim(x_min, x_max)
        axis_L2_error.set_ylim(y_min, y_max)
        axis_L2_error.set_aspect("equal")
        axis_L2_error.set_title(
            r"Relative L2 error = {:.4e}".format(relative_L2_error.item())
        )
        figure_error.colorbar(l2_error_surface, ax=axis_L2_error)
        h1_error_surface = PolyCollection(
            coordinates_4_triangles[..., :3, :],  # type: ignore
            array=H1_error.sqrt().squeeze(-1).numpy(force=True),
            cmap="viridis",
            edgecolor="black",
            linewidths=0.2,
        )
        axis_H1_error.add_collection(h1_error_surface)
        axis_H1_error.set_xlim(x_min, x_max)
        axis_H1_error.set_ylim(y_min, y_max)
        axis_H1_error.set_aspect("equal")
        axis_H1_error.set_title(
            r"Relative H1 error = {:.4e}".format(relative_H1_error.item())
        )
        figure_error.colorbar(h1_error_surface, ax=axis_H1_error)
        figure_error.tight_layout()

        return figure_solution, figure_error

    def plot(self, numerical_solution: torch.Tensor) -> Tuple[Figure, ...]:
        """Plot the numerical solution over the mesh."""
        return self._plot(numerical_solution)

    @abc.abstractmethod
    def compute_error(self, numerical_solution: torch.Tensor):
        """Compute the error between numerical and exact solution."""
        raise NotImplementedError

    @abc.abstractmethod
    def solve(self) -> torch.Tensor:
        """compute the numerical solution."""
        raise NotImplementedError
