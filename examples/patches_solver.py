import triangle
import torch
import matplotlib.pyplot as plt
from torch_fem import (
    MeshTri,
    PatchesSolver as Solver,
    ExponentialProblem as Problem,
    FeedForwardNeuralNetwork as NeuralNetwork,
    DistanceFunctionBC,
    Patches,
)

# torch.set_default_dtype(torch.float64)


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


def generate_patches_info(n):
    """generates a set of centers and radius"""
    initial_centers = [(0.5, 0.5)]
    initial_radius = [0.5]

    for _ in range(n):
        new_centers = []
        new_radius = []
        for (cx, cy), r in zip(initial_centers, initial_radius):
            new_r = r / 2
            new_centers.extend(
                [
                    (cx - new_r, cy - new_r),
                    (cx - new_r, cy + new_r),
                    (cx + new_r, cy - new_r),
                    (cx + new_r, cy + new_r),
                ]
            )
            new_radius.extend([new_r] * 4)
        initial_centers, initial_radius = new_centers, new_radius

    return torch.Tensor(initial_centers), torch.Tensor(initial_radius).unsqueeze(-1)


centers, radius = generate_patches_info(3)

patches = Patches(centers=centers, radius=radius)

neural_network = NeuralNetwork(
    input_dimension=2,
    output_dimension=1,
    nb_hidden_layers=5,
    neurons_per_layers=15,
    boundary_condition_modifier=DistanceFunctionBC(segments),
)


solver = Solver(
    mesh=patches,
    p_order=P_ORDER,
    q_order=2 * P_ORDER,
    problem=Problem(),
    neural_network=neural_network,
    error_mesh=mesh,
    optimizer=torch.optim.Adam,
    jit_compile=True,
    posteriori_error=False,
    epochs=10000,
    use_early_stopping=True,
    early_stopping_patience=1000,
    min_delta=1e-15,
)


solution = solver.solve()


solver.plot(solution)

plt.show()
