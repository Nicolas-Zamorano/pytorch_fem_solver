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

    def dirichlet_boundary(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Dirichlet boundary condition function."""
        raise NotImplementedError

    def precomputed_H1_norm(
        self,
        _,
        value: torch.Tensor,
        value_dx: torch.Tensor,
        value_dy: torch.Tensor,
        value_dz: torch.Tensor = torch.tensor([0.0]),
    ) -> torch.Tensor:
        """H1 norm for precomputed values."""
        return value**2 + value_dx**2 + value_dy**2 + value_dz**2

    def precomputed_L2_norm(
        self,
        _,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """L2 norm for precomputed values."""
        return value**2

    def H1_norm(
        self, basis, function, function_dx, function_dy, function_dz=None
    ) -> torch.Tensor:
        """H1 norm."""
        function_value = function(basis.integration_points)
        function_dx_value = function_dx(basis.integration_points)
        function_dy_value = function_dy(basis.integration_points)

        if function_dz is not None:
            function_dz_value = function_dz(basis.integration_points)
        else:
            function_dz_value = 0.0

        return (
            function_value**2
            + function_dx_value**2
            + function_dy_value**2
            + function_dz_value**2
        )

    def L2_norm(self, basis, function) -> torch.Tensor:
        """L2 norm."""
        function_value = function(basis.integration_points)

        return function_value**2

    @staticmethod
    @abc.abstractmethod
    def bilinear_form(basis) -> torch.Tensor:
        """Bilinear form of the PDE."""
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def linear_form(basis, rhs_value: torch.Tensor) -> torch.Tensor:
        """Linear form of the PDE."""
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def residual(
        basis, gradient: torch.Tensor, rhs_value: torch.Tensor
    ) -> torch.Tensor:
        """Residual form of the PDE."""
        raise NotImplementedError

    @staticmethod
    def bulk_residual(
        basis, laplacian: torch.Tensor, rhs_value: torch.Tensor
    ) -> torch.Tensor:
        """Bulk residual of the PDE."""
        raise NotImplementedError

    @staticmethod
    def jump_residual(
        edges_basis, gradient_for_jump: torch.Tensor, normals_4_elements: torch.Tensor
    ) -> torch.Tensor:
        """Jump residual of the PDE."""
        raise NotImplementedError
