import abc
import torch
from ..problems import AbstractProblem
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection


class AbstractSolver(abc.ABC):
    """Abstract base class for solvers."""

    def __init__(
        self,
        mesh,
        p_order: int,
        q_order: int,
        problem: AbstractProblem,
    ):
        self.mesh = mesh
        self.p_order = p_order
        self.q_order = q_order
        self.problem = problem
        self.precomputed_values = self.precompute_values()

    @abc.abstractmethod
    def precompute_values(self) -> dict:
        raise NotImplementedError

    @abc.abstractmethod
    def compute_error(self, numerical_solution: torch.Tensor):
        """Compute the error between numerical and exact solution."""
        raise NotImplementedError

    @abc.abstractmethod
    def solve(self):
        """compute the numerical solution."""
        raise NotImplementedError

    @abc.abstractmethod
    def plot(self, numerical_solution: torch.Tensor):
        """Plot the numerical solution over the mesh."""
        raise NotImplementedError

    # @property
    # @abc.abstractmethod
    # def basis(self) -> AbstractBasis:
    #     """Finite element basis associated with the solver."""
    #     raise NotImplementedError

    # @property
    # @abc.abstractmethod
    # def elements(self) -> AbstractElement:
    #     """Finite element associated with the solver."""
    #     raise NotImplementedError

    def convernge(self, nb_refiments: int):
        """Compute convergence study over a number of mesh refinements."""
