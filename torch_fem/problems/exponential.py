import torch
from .poisson_problem import PoissonProblem


class ExponentialProblem(PoissonProblem):
    """Exponential PDE problem definition."""

    def __init__(self, scaling_constant=1.0, exponential_coefficient=5.0):
        self.scaling_constant = scaling_constant
        self.exponential_coefficient = exponential_coefficient

    def exact(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Exact solution of the PDE."""
        x, y = torch.split(coordinates, 1, -1)
        return (
            self.scaling_constant
            * x
            * y
            * (1 - x)
            * (1 - y)
            * (torch.exp(self.exponential_coefficient * x) - 1)
        )

    def exact_dx(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Exact solution derivative with respect to x."""

        x, y = torch.split(coordinates, 1, -1)
        exponential_value = torch.exp(self.exponential_coefficient * x)
        return (
            self.scaling_constant
            * y
            * (1 - y)
            * (
                (1 - 2 * x) * (exponential_value - 1)
                + self.exponential_coefficient * x * (1 - x) * exponential_value
            )
        )

    def exact_dy(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Exact solution derivative with respect to y."""
        x, y = torch.split(coordinates, 1, -1)
        exponential_value = torch.exp(self.exponential_coefficient * x)

        return (
            self.scaling_constant * (1 - 2 * y) * x * (1 - x) * (exponential_value - 1)
        )

    def rhs(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Right-hand side function."""
        x, y = torch.split(coordinates, 1, -1)

        exponential_value = torch.exp(self.exponential_coefficient * x)

        fxx = (
            self.scaling_constant
            * y
            * (1 - y)
            * (
                -2 * (exponential_value - 1)
                + 2 * self.exponential_coefficient * (1 - 2 * x) * exponential_value
                + self.exponential_coefficient**2 * x * (1 - x) * exponential_value
            )
        )

        fyy = self.scaling_constant * (-2) * x * (1 - x) * (exponential_value - 1)

        lap = fxx + fyy
        return -lap

    def dirichlet_boundary(self, coordinates: torch.Tensor) -> torch.Tensor:
        return self.exact(coordinates)
