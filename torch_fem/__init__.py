"""Initiation for creating a package"""

from .basis import (
    Basis,
    FractureBasis,
    InteriorEdgesBasis,
    InteriorEdgesFractureBasis,
    PatchesBasis,
)
from .element import ElementLine, ElementTri
from .mesh import FracturesTri, MeshTri, Patches
from .model import Model, FeedForwardNeuralNetwork, DistanceFunctionBC
from .solvers import FemSolver, VPINNsSolver, RVPINNsSolver, FEINNsSolver
from .problems import (
    ExponentialProblem,
    SinsProblem,
    TanhProblem,
    PoissonProblem,
    FractureProblem,
)


__all__ = [
    "Basis",
    "FractureBasis",
    "InteriorEdgesBasis",
    "InteriorEdgesFractureBasis",
    "PatchesBasis",
    "ElementLine",
    "ElementTri",
    "FracturesTri",
    "MeshTri",
    "Patches",
    "Model",
    "FeedForwardNeuralNetwork",
    "DistanceFunctionBC",
    "FemSolver",
    "ExponentialProblem",
    "SinsProblem",
    "TanhProblem",
    "PoissonProblem",
    "FractureProblem",
    "VPINNsSolver",
    "RVPINNsSolver",
    "FEINNsSolver",
]
