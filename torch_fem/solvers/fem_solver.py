from typing import Tuple
import torch
from .abstract_solver import AbstractSolver
from torch_fem import Basis, ElementTri


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

        boundary_value = self.problem.exact(self.basis.coords_4_global_dofs)[
            self.basis.basis_parameters["boundary_dofs"]
        ]

        solution_tensor = self.basis.solution_tensor()
        solution_tensor[self.basis.basis_parameters["boundary_dofs"]] = boundary_value

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
