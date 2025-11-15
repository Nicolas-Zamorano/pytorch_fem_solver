import torch
from .poisson_problem import PoissonProblem


class TanhProblem(PoissonProblem):
    """Tanh PDE problem definition."""

    def exact(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Exact solution of the PDE."""
        x, y = torch.split(coordinates, 1, -1)
        return torch.tanh(2 * (x**3 - y**4))

    def exact_dx(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Exact solution derivative with respect to x."""

        x, y = torch.split(coordinates, 1, -1)
        return 6 * x**2 * (1.0 / torch.cosh(2 * (x**3 - y**4)) ** 2)

    def exact_dy(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Exact solution derivative with respect to y."""
        x, y = torch.split(coordinates, 1, -1)

        return -8 * y**3 * (1.0 / torch.cosh(2 * (x**3 - y**4)) ** 2)

    def rhs(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Right-hand side function."""
        x, y = torch.split(coordinates, 1, -1)

        return (
            4
            * (1.0 / torch.cosh(2 * (x**3 - y**4)) ** 2)
            * (
                -3 * x
                + 6 * y**2
                + 2 * (9 * x**4 + 16 * y**6) * torch.tanh(2 * (x**3 - y**4))
            )
        )
