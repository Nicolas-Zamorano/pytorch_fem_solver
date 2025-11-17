import torch
import abc


class AbstractProblem(abc.ABC):
    """Abstract base class for defining PDE problems."""

    @abc.abstractmethod
    def rhs(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Right-hand side function."""
        raise NotImplementedError

    def exact(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Exact solution of the PDE."""
        raise NotImplementedError

    def exact_dx(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Exact solution derivative with respect to x."""
        raise NotImplementedError

    def exact_dy(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Exact solution derivative with respect to y."""
        raise NotImplementedError

    def exact_dz(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Exact solution derivative with respect to z."""
        raise NotImplementedError

    def precomputed_H1_norm(
        self,
        _,
        value: torch.Tensor,
        value_dx: torch.Tensor,
        value_dy: torch.Tensor,
    ) -> torch.Tensor:
        """H1 norm for precomputed values."""
        return value**2 + value_dx**2 + value_dy**2

    def precomputed_L2_norm(
        self,
        _,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """L2 norm for precomputed values."""
        return value**2

    def H1_norm(self, basis, function, function_dx, function_dy) -> torch.Tensor:
        """H1 norm."""
        function_value = function(basis.integration_points)
        function_dx_value = function_dx(basis.integration_points)
        function_dy_value = function_dy(basis.integration_points)

        return function_value**2 + function_dx_value**2 + function_dy_value**2

    def L2_norm(self, basis, function) -> torch.Tensor:
        """L2 norm."""
        function_value = function(basis.integration_points)

        return function_value**2

    @staticmethod
    @abc.abstractmethod
    def bilinear_form(basis) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def linear_form(basis, rhs_values: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def residual(
        basis, gradient: torch.Tensor, rhs_values: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def bulk_residual(
        basis, laplcian: torch.Tensor, rhs_values: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def jump_residual(
        edges_basis, gradient_for_jump: torch.Tensor, normals_4_elements: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def dirichlet_boundary(coordinates: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
