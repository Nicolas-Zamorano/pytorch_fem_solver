from math import pi
import torch
from .poisson_problem import PoissonProblem


class SinsProblem(PoissonProblem):
    """Sine PDE problem definition."""

    def exact(self, coordinates: torch.Tensor) -> torch.Tensor:
        x, y = torch.split(coordinates, 1, -1)
        return torch.sin(pi * x) * torch.sin(pi * y)

    def exact_dx(self, coordinates: torch.Tensor) -> torch.Tensor:
        x, y = torch.split(coordinates, 1, -1)
        return pi * torch.cos(pi * x) * torch.sin(pi * y)

    def exact_dy(self, coordinates: torch.Tensor) -> torch.Tensor:
        x, y = torch.split(coordinates, 1, -1)
        return pi * torch.sin(pi * x) * torch.cos(pi * y)

    def rhs(self, coordinates: torch.Tensor) -> torch.Tensor:
        x, y = torch.split(coordinates, 1, -1)
        return 2 * pi**2 * torch.sin(pi * x) * torch.sin(pi * y)
