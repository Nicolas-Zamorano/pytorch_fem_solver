import torch

import matplotlib.pyplot as plt

from Neural_Network import Neural_Network
from fem import Patches, Elements, Basis

centers = torch.tensor([[0.5, 0.5]])
radius = torch.tensor([[0.5]])

mesh = Patches(centers, radius)

mesh.uniform_refine(1)

elements = Elements(P_order = 1, 
                    int_order = 2)

V = Basis(mesh, elements)


print("ez")



