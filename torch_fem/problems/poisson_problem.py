import torch
from .abstract_problem import AbstractProblem


class PoissonProblem(AbstractProblem):
    """Poisson problem definition."""

    @staticmethod
    def bilinear_form(basis) -> torch.Tensor:
        """Bilinear form."""
        return basis.v_grad @ basis.v_grad.mT

    @staticmethod
    def linear_form(basis, rhs_values: torch.Tensor) -> torch.Tensor:
        """Linear form for the right-hand side."""
        return rhs_values * basis.v

    @staticmethod
    def residual(
        basis, gradient: torch.Tensor, rhs_values: torch.Tensor
    ) -> torch.Tensor:
        """Residual form."""
        return basis.v_grad @ gradient.mT - rhs_values * basis.v

    @staticmethod
    def bulk_residual(
        _,
        laplacian: torch.Tensor,
        rhs_values: torch.Tensor,
    ) -> torch.Tensor:
        """Bulk residual form."""
        return (laplacian + rhs_values) ** 2

    @staticmethod
    def jump_residual(
        _,
        gradient_for_jump: torch.Tensor,
        normals_4_elements: torch.Tensor,
    ) -> torch.Tensor:
        """Jump residual form."""
        gradient_minus, gradient_plus = torch.unbind(gradient_for_jump, dim=-4)

        return ((gradient_plus - gradient_minus) * normals_4_elements).sum(
            -1, keepdim=True
        ) ** 2
