import os
from typing import Tuple
import torch
from .abstract_solver import AbstractSolver


class FemSolver(AbstractSolver):
    """Finite Element Method (FEM) solver."""

    def precompute_values(
        self, mesh, polynomial_order, integral_order, error_mesh=None
    ):
        basis, precomputed_values, error_basis = self._precompute_values(
            mesh, polynomial_order, integral_order, error_mesh
        )

        solution_tensor = basis.evaluate_at_boundary(self.problem.dirichlet_boundary)
        precomputed_values["solution_tensor"] = solution_tensor

        return basis, precomputed_values, error_basis

    def solve(self) -> torch.Tensor:
        stiffness_matrix = self.basis.integrate_bilinear_form(
            self.problem.bilinear_form
        )
        load_vector = self.basis.integrate_linear_form(
            self.problem.linear_form, rhs_value=self.precomputed_values["rhs_values"]
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
