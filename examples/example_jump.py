"""Example of using jump residual for a problem with discontinuous solution"""

import math

import triangle as tr
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import torch

from torch_fem import (
    Basis,
    ElementLine,
    ElementTri,
    InteriorEdgesBasis,
    MeshTri,
    Model,
    FeedForwardNeuralNetwork as NeuralNetwork,
)

# torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
# torch.cuda.empty_cache()
torch.set_default_dtype(torch.float64)


class BoundaryConstrain(torch.nn.Module):
    """Class to strongly apply bc"""

    def forward(self, inputs):
        """Boundary condition modifier function."""
        x, y = torch.split(inputs, 1, dim=-1)
        return x * (x - 1) * y * (y - 1)


NN = NeuralNetwork(
    input_dimension=2,
    output_dimension=1,
    nb_hidden_layers=5,
    neurons_per_layers=25,
    boundary_condition_modifier=BoundaryConstrain(),
)
# ---------------------- FEM Parameters ----------------------#

mesh_data = tr.triangulate(
    {"vertices": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]},
    "Dqena" + str(0.5**8),
)

mesh = MeshTri(triangulation=mesh_data)

elements = ElementTri(polynomial_order=1, integration_order=2)

elements_1D = ElementLine(polynomial_order=1, integration_order=2)

V_edges = InteriorEdgesBasis(mesh, elements_1D)

V = Basis(mesh, elements)

_, interpolator_to_edges_grad = V.interpolate(V_edges)

_, interpolators_grad = V.interpolate(V)

# ---------------------- Residual Parameters ----------------------#


def rhs(x, y):
    """Right-hand side function"""
    return 2.0 * math.pi**2 * torch.sin(math.pi * x) * torch.sin(math.pi * y)


h_T = V.mesh["cells", "length"]
h_E = V.mesh["interior_edges", "length"].unsqueeze(-2)
n_E = V.mesh["interior_edges", "normals"].unsqueeze(-2)


def jump(_, normal_elements, edge_size, nn):
    """Jump term for discontinuous solutions"""
    interpolator_u_grad_plus, interpolator_u_grad_minus = torch.unbind(
        interpolator_to_edges_grad(nn), dim=-4
    )
    return (
        edge_size
        * (
            (interpolator_u_grad_plus * normal_elements).sum(-1, keepdim=True)
            + (interpolator_u_grad_minus * -normal_elements).sum(-1, keepdim=True)
        )
        ** 2
    )


def rhs_term(basis, triangle_size, nn):
    """Residual term for the right-hand side"""
    x, y = torch.split(basis.integration_points, 1, dim=-1)

    return triangle_size**2 * (rhs(x, y) + nn.laplacian(basis.integration_points)) ** 2


# ---------------------- Error Parameters ----------------------#


def exact(x, y):
    """Exact solution of the PDE."""
    return torch.sin(math.pi * x) * torch.sin(math.pi * y)


def exact_dx(x, y):
    """Exact solution derivative with respect to x."""
    return math.pi * torch.cos(math.pi * x) * torch.sin(math.pi * y)


def exact_dy(x, y):
    """Exact solution derivative with respect to y."""
    return math.pi * torch.sin(math.pi * x) * torch.cos(math.pi * y)


def h1_exact(basis):
    """H1 norm of the exact solution."""
    x, y = torch.split(basis.integration_points, 1, dim=-1)

    return exact(x, y) ** 2 + exact_dx(x, y) ** 2 + exact_dy(x, y) ** 2


def h1_norm(basis, neural_network, gradient):
    """H1 norm of the neural network solution."""
    integration_points = basis.integration_points
    x, y = torch.split(integration_points, 1, dim=-1)

    nn_dx, nn_dy = torch.split(gradient(integration_points), 1, dim=-1)

    return (
        (exact(x, y) - neural_network(integration_points)) ** 2
        + (exact_dx(x, y) - nn_dx) ** 2
        + (exact_dy(x, y) - nn_dy) ** 2
    )


exact_norm = torch.sqrt(torch.sum(V.integrate_functional(h1_exact)))

# ---------------------- Training ----------------------#

jump_history = []
bulk_history = []


def training_step(neural_network):
    """Training step for the neural network."""

    jump_term = V_edges.integrate_functional(jump, n_E, h_E, neural_network)

    bulk_term = V.integrate_functional(rhs_term, h_T, neural_network)

    loss_value = torch.sum(bulk_term) + torch.sum(jump_term)

    relative_loss = torch.sqrt(loss_value) / exact_norm**2

    h1_error = torch.sqrt(
        torch.sum(
            V.integrate_functional(h1_norm, neural_network, neural_network.gradient)
        )
    )

    jump_history.append(torch.sqrt(torch.sum(jump_term)).item())
    bulk_history.append(torch.sqrt(torch.sum(bulk_term)).item())

    return loss_value, relative_loss, h1_error / exact_norm


model = Model(
    neural_network=NN,
    training_step=training_step,
    epochs=2000,
    optimizer=torch.optim.Adam,
    optimizer_kwargs={"lr": 0.001},
    # learning_rate_scheduler=torch.optim.lr_scheduler.ExponentialLR,
    # scheduler_kwargs={"gamma": 0.9},
    use_early_stopping=False,
    early_stopping_patience=25,
    min_delta=1e-16,
)

model.train()

# ---------------------- Plotting ----------------------#

model.load_optimal_parameters()

h1_error_plot = (
    torch.sqrt(V.integrate_functional(h1_norm, NN, NN.gradient))
    .squeeze(-1)
    .numpy(force=True)
)

figure_solution, axis_solution = plt.subplots()

c4e = torch.Tensor.numpy(V.mesh["cells", "coordinates"], force=True)

collection = PolyCollection(
    c4e,  # type: ignore
    array=h1_error_plot,
    cmap="viridis",
    edgecolors="black",
    linewidths=0.2,
)

axis_solution.add_collection(collection)
axis_solution.autoscale_view()

axis_solution.set_xlabel("x")
axis_solution.set_ylabel("y")
color_bar = plt.colorbar(collection, ax=axis_solution)
color_bar.set_label(r"$H^1$ error")

figure_solution.tight_layout()

model.plot_training_history(
    plot_names={
        "loss": r"$\mathcal{L}(u_{\theta})$",
        "validation": r"$\frac{\sqrt{\mathcal{L}(u_{\theta})}}{\|u\|_U}$",
        "accuracy": r"$\frac{\|u-u_{\theta}\|_U}{\|u_{\theta}\|_U}$",
        "title": "only a posteriori estimator",
    }
)

fig_errors, ax_errors = plt.subplots()
ax_errors.semilogy(jump_history, label="Jump term")
ax_errors.semilogy(bulk_history, label="Bulk term")
ax_errors.set_xlabel("Epochs")
ax_errors.set_ylabel("Error terms")
ax_errors.legend()

plt.show()
