import triangle
import torch
import matplotlib.pyplot as plt
from torch_fem import (
    MeshTri,
    FEINNsSolver as Solver,
    SinsProblem as Problem,
    FeedForwardNeuralNetwork as NeuralNetwork,
    DistanceFunctionBC,
)

torch.set_default_dtype(torch.float64)


mesh_data = triangle.triangulate(
    {"vertices": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]},
    "qena" + str(0.5**6),
)

mesh = MeshTri(triangulation=mesh_data)

P_ORDER = 2

segments = torch.tensor(
    [
        [[0.0, 0.0], [1.0, 0.0]],
        [[1.0, 0.0], [1.0, 1.0]],
        [[1.0, 1.0], [0.0, 1.0]],
        [[0.0, 1.0], [0.0, 0.0]],
    ]
)

neural_network = NeuralNetwork(
    input_dimension=2,
    output_dimension=1,
    nb_hidden_layers=5,
    neurons_per_layers=50,
    # boundary_condition_modifier=DistanceFunctionBC(segments),
)


solver = Solver(
    mesh=mesh,
    p_order=P_ORDER,
    q_order=2 * P_ORDER,
    problem=Problem(),
    neural_network=neural_network,
    jit_compile=True,
    posteriori_error=True,
    epochs=25000,
    use_early_stopping=False,
    early_stopping_patience=150,
    min_delta=1e-15,
)


solution = solver.solve()


solver.plot(solution)

plt.show()
