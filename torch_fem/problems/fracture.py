import torch
from .poisson_problem import PoissonProblem


class FractureProblem(PoissonProblem):
    """Fracture PDE problem definition."""

    def exact(self, coordinates):
        """Exact solution."""
        x, y, z = torch.split(coordinates, 1, dim=-1)
        x_fracture_1, _ = torch.split(x, 1, dim=0)
        y_fracture_1, y_fracture_2 = torch.split(y, 1, dim=0)
        _, z_fracture_2 = torch.split(z, 1, dim=0)

        exact_fracture_1 = (
            -y_fracture_1
            * (1 - y_fracture_1)
            * torch.abs(x_fracture_1)
            * (x_fracture_1**2 - 1)
        )
        exact_fracture_2 = (
            y_fracture_2
            * (1 - y_fracture_2)
            * torch.abs(z_fracture_2)
            * (z_fracture_2**2 - 1)
        )

        exact_value = torch.cat([exact_fracture_1, exact_fracture_2], dim=0)

        return exact_value

    def exact_dx(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Exact solution derivative with respect to x."""
        x, y, z = torch.split(coordinates, 1, dim=-1)
        x_fracture_1, _ = torch.split(x, 1, dim=0)

        exact_dx_fracture_1 = (
            -y
            * (1 - y)
            * (
                torch.sign(x_fracture_1) * (x_fracture_1**2 - 1)
                + 2 * x_fracture_1 * torch.abs(x_fracture_1)
            )
        )
        exact_dx_fracture_2 = torch.zeros_like(z)

        exact_dx_value = torch.cat([exact_dx_fracture_1, exact_dx_fracture_2], dim=0)

        return exact_dx_value

    def exact_dy(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Exact solution derivative with respect to y."""
        x, y, z = torch.split(coordinates, 1, dim=-1)
        x_fracture_1, _ = torch.split(x, 1, dim=0)
        y_fracture_1, y_fracture_2 = torch.split(y, 1, dim=0)
        _, z_fracture_2 = torch.split(z, 1, dim=0)

        exact_dy_fracture_1 = (
            -(1 - 2 * y_fracture_1) * torch.abs(x_fracture_1) * (x_fracture_1**2 - 1)
        )
        exact_dy_fracture_2 = (
            (1 - 2 * y_fracture_2) * torch.abs(z_fracture_2) * (z_fracture_2**2 - 1)
        )

        exact_dy_value = torch.cat([exact_dy_fracture_1, exact_dy_fracture_2], dim=0)

        return exact_dy_value

    def exact_dz(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Exact solution derivative with respect to z."""
        x, y, z = torch.split(coordinates, 1, dim=-1)
        _, z_fracture_2 = torch.split(z, 1, dim=0)

        exact_dz_fracture_1 = torch.zeros_like(x)
        exact_dz_fracture_2 = (
            y
            * (1 - y)
            * (
                torch.sign(z_fracture_2) * (z_fracture_2**2 - 1)
                + 2 * z_fracture_2 * torch.abs(z_fracture_2)
            )
        )

        exact_dz_value = torch.cat([exact_dz_fracture_1, exact_dz_fracture_2], dim=0)

        return exact_dz_value

    def rhs(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Right-hand side function."""
        x, y, z = torch.split(coordinates, 1, dim=-1)
        x_fracture_1, _ = torch.split(x, 1, dim=0)
        y_fracture_1, y_fracture_2 = torch.split(y, 1, dim=0)
        _, z_fracture_2 = torch.split(z, 1, dim=0)

        rhs_fracture_1 = 6.0 * (y_fracture_1 - y_fracture_1**2) * torch.abs(
            x_fracture_1
        ) - 2.0 * (torch.abs(x_fracture_1) ** 3 - torch.abs(x_fracture_1))
        rhs_fracture_2 = -6.0 * (y_fracture_2 - y_fracture_2**2) * torch.abs(
            z_fracture_2
        ) + 2.0 * (torch.abs(z_fracture_2) ** 3 - torch.abs(z_fracture_2))

        rhs_value = torch.cat([rhs_fracture_1, rhs_fracture_2], dim=0)

        return rhs_value

    def precomputed_H1_norm(
        self,
        _,
        value: torch.Tensor,
        value_dx: torch.Tensor,
        value_dy: torch.Tensor,
        value_dz: torch.Tensor,
    ) -> torch.Tensor:
        return value**2 + value_dx**2 + value_dy**2 + value_dz**2

    # def exact_grad(coordinates: torch.Tensor) -> torch.Tensor:
    #     """Gradient of the exact solution."""
    #     x, y, z = torch.split(coordinates, 1, dim=-1)
    #     x_fracture_1, _ = torch.split(x, 1, dim=0)
    #     y_fracture_1, y_fracture_2 = torch.split(y, 1, dim=0)
    #     _, z_fracture_2 = torch.split(z, 1, dim=0)

    #     exact_dx_fracture_1 = (
    #         -y_fracture_1
    #         * (1 - y_fracture_1)
    #         * (
    #             torch.sign(x_fracture_1) * (x_fracture_1**2 - 1)
    #             + 2 * x_fracture_1 * torch.abs(x_fracture_1)
    #         )
    #     )
    #     exact_dy_fracture_1 = (
    #         -(1 - 2 * y_fracture_1) * torch.abs(x_fracture_1) * (x_fracture_1**2 - 1)
    #     )
    #     exact_dz_fracture_1 = torch.zeros_like(exact_dx_fracture_1)

    #     exact_grad_fracture_1 = torch.cat(
    #         [exact_dx_fracture_1, exact_dy_fracture_1, exact_dz_fracture_1], dim=-1
    #     )

    #     exact_dy_fracture_2 = (
    #         (1 - 2 * y_fracture_2) * torch.abs(z_fracture_2) * (z_fracture_2**2 - 1)
    #     )
    #     exact_dz_fracture_2 = (
    #         y_fracture_2
    #         * (1 - y_fracture_2)
    #         * (
    #             torch.sign(z_fracture_2) * (z_fracture_2**2 - 1)
    #             + 2 * z_fracture_2 * torch.abs(z_fracture_2)
    #         )
    #     )
    #     exact_dx_fracture_2 = torch.zeros_like(exact_dz_fracture_2)

    #     exact_grad_fracture_2 = torch.cat(
    #         [exact_dx_fracture_2, exact_dy_fracture_2, exact_dz_fracture_2], dim=-1
    #     )

    #     grad_value = torch.cat([exact_grad_fracture_1, exact_grad_fracture_2], dim=0)

    #     return grad_value
