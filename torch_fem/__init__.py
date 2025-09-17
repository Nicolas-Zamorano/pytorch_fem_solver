"""Initiation for creating a package"""

from .basis import Basis, FractureBasis, InteriorEdgesBasis, InteriorEdgesFractureBasis
from .element import ElementLine, ElementTri
from .mesh import FracturesTri, MeshTri
from .model import Model, FeedForwardNeuralNetwork
