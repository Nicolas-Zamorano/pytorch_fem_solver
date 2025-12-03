import triangle
import torch
import matplotlib.pyplot as plt
from torch_fem import (
    MeshTri,
    SimplePatchesSolver as Solver,
    ExponentialProblem as Problem,
    FeedForwardNeuralNetwork as NeuralNetwork,
    DistanceFunctionBC,
)

torch.set_default_dtype(torch.float32)


mesh_data = triangle.triangulate(
    {"vertices": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]},
    "qena" + str(0.5**8),
)

mesh = MeshTri(triangulation=mesh_data)

P_ORDER = 1

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
    nb_hidden_layers=2,
    neurons_per_layers=8,
    boundary_condition_modifier=DistanceFunctionBC(segments),
)

solver = Solver(
    mesh=mesh,
    p_order=P_ORDER,
    q_order=2 * P_ORDER,
    problem=Problem(),
    neural_network=neural_network,
    optimizer=torch.optim.Adam,
    optimizer_kwargs={
        "lr": 1e-3,
    },
    jit_compile=True,
    posteriori_error=False,
    epochs=10000,
    use_early_stopping=True,
    early_stopping_patience=100,
    min_delta=1e-15,
)

solution = solver.solve()

solver.plot(solution)

fig_errors, (ax_errors, ax_efectivity) = plt.subplots(1, 2, figsize=(10, 4))
ax_errors.plot(solver.h1_error_sum, label=r"$\sum_{i\in I_p}\|u-u_\theta\|_{H^1(P_i)}$")
ax_errors.plot(solver.h1_error, label=r"$\|u-u_\theta\|_{H^1(\Omega)}$")
ax_errors.legend()
ax_errors.set_yscale("log")

effectivity = [
    h1_s / h1_e if h1_s != 0 else 0
    for h1_e, h1_s in zip(solver.h1_error, solver.h1_error_sum)
]

ax_efectivity.plot(effectivity, label="Effectivity Index")
ax_efectivity.set_yscale("log")
ax_efectivity.legend()

plt.show()
