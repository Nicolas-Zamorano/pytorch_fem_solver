import triangle
import torch
import matplotlib.pyplot as plt
from torch_fem import (
    MeshTri,
    FemSolver as Solver,
    ExponentialProblem as Problem,
)


torch.set_default_dtype(torch.float64)

mesh_data = triangle.triangulate(
    {"vertices": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]},
    "qena" + str(0.5**8),
)

mesh = MeshTri(triangulation=mesh_data)

P_ORDER = 1

solver = Solver(
    mesh=mesh, polynomial_order=P_ORDER, integral_order=2 * P_ORDER, problem=Problem()
)

solution = solver.solve()

figure_solution, figure_error = solver.plot(solution)

plt.show()
