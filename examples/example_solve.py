import triangle
import torch
from torch_fem import (
    MeshTri,
    FemSolver as Solver,
    SinsProblem as Problem,
    FeedForwardNeuralNetwork as NeuralNetwork,
    DistanceFunctionBC,
)


mesh_data = triangle.triangulate(
    {"vertices": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]},
    "qena" + str(0.5**8),
)

mesh = MeshTri(triangulation=mesh_data)


# class BoundaryConstrain(torch.nn.Module):
#     """Class to strongly apply bc"""

#     def forward(self, inputs: torch.Tensor) -> torch.Tensor:
#         """Boundary condition modifier function."""
#         x, y = torch.split(inputs, 1, dim=-1)
#         return x * (x - 1) * y * (y - 1)


segments = torch.tensor(
    [
        [[0.0, 0.0], [1.0, 0.0]],
        [[1.0, 0.0], [1.0, 1.0]],
        [[1.0, 1.0], [0.0, 1.0]],
        [[0.0, 1.0], [0.0, 0.0]],
    ]
)

# nn = NeuralNetwork(
#     input_dimension=2,
#     output_dimension=1,
#     nb_hidden_layers=5,
#     neurons_per_layers=25,
#     boundary_condition_modifier=DistanceFunctionBC(segments),
#     # boundary_condition_modifier=BoundaryConstrain(),
# )


# solver = Solver(
#     mesh=mesh,
#     p_order=3,
#     q_order=6,
#     problem=Problem(),
#     neural_network=nn,
#     # jit_compile=False,
#     # posteriori_error=False,
#     epochs=15000,
# )

solver = Solver(mesh=mesh, p_order=3, q_order=6, problem=Problem())

solution = solver.solve()

solver.plot(solution)
