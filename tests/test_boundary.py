import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import torch
import triangle as tr

from torch_fem import (
    Basis,
    ElementTri,
    MeshTri,
    FeedForwardNeuralNetwork as NeuralNetwork,
)


# torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
# torch.cuda.empty_cache()
torch.set_default_dtype(torch.float64)

# ---------------------- FEM Parameters ----------------------#

mesh_data = tr.triangulate(
    {"vertices": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]},
    "Dqena" + str(0.5**10),
)

mesh = MeshTri(triangulation=mesh_data)

elements = ElementTri(polynomial_order=3, integration_order=6)

discrete_basis = Basis(mesh, elements)


integration_points = discrete_basis.integration_points

# Precompute values

exact = lambda x: torch.ones_like(x[..., :1])

exact_value = exact(integration_points)

value_boundary_condition = discrete_basis.solution_tensor()
value_boundary_condition[discrete_basis.basis_parameters["boundary_dofs"], :] += exact(
    discrete_basis.coords4global_dofs
)[discrete_basis.basis_parameters["boundary_dofs"], :]

extended_boundary_value, _ = discrete_basis.interpolate(
    discrete_basis, value_boundary_condition
)

# ---------------------- Neural Network Parameters ----------------------#


class BoundaryConstrain(torch.nn.Module):
    """Class to strongly apply bc"""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Boundary condition modifier function."""
        x, y = torch.split(inputs, 1, dim=-1)
        return x * (x - 1) * y * (y - 1)


segments = torch.tensor(
    [
        [[0.0, 0.0], [1.0, 0.0]],
        [[1.0, 0.0], [1.0, 1.0]],
        [[1.0, 1.0], [0.0, 1.0]],
        [[0.0, 1.0], [0.0, 0.0]],
    ]
)


NN = NeuralNetwork(
    input_dimension=2,
    output_dimension=1,
    nb_hidden_layers=3,
    neurons_per_layers=50,
    # boundary_condition_modifier=DistanceFunctionBC(segments),
    boundary_condition_modifier=BoundaryConstrain(),
    use_xavier_initialization=True,
    boundary_condition_value=extended_boundary_value,
)

value = NN(integration_points)

figure_solution, axis_solution = plt.subplots()

c4e = torch.Tensor.numpy(discrete_basis.mesh["cells", "coordinates"], force=True)

triangles_plot = PolyCollection(
    c4e,  # type: ignore
    array=extended_boundary_value.mean(1).reshape(-1),
    cmap="viridis",
    edgecolors="black",
    linewidths=0.2,
)

axis_solution.add_collection(triangles_plot)
axis_solution.autoscale_view()

axis_solution.set_xlabel("x")
axis_solution.set_ylabel("y")
axis_solution.set_xlim((0, 1))
axis_solution.set_ylim((0, 1))
color_bar = plt.colorbar(triangles_plot, ax=axis_solution)
color_bar.set_label(r"$H^1$ error")

figure_solution.tight_layout()

plt.show()
