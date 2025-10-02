"# Example of solving a Poisson equation using a neural network and FEM basis functions."

import math

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import torch
import triangle as tr

from torch_fem import (
    Basis,
    ElementTri,
    MeshTri,
    ElementLine,
    InteriorEdgesBasis,
    FeedForwardNeuralNetwork as NeuralNetwork,
    Model,
)

# torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
# torch.cuda.empty_cache()
torch.set_default_dtype(torch.float64)


# ---------------------- Neural Network Parameters ----------------------#


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

discrete_basis = Basis(mesh, elements)

elements_1D = ElementLine(polynomial_order=1, integration_order=2)

V_edges = InteriorEdgesBasis(mesh, elements_1D)

_, interpolator_to_edges_grad = discrete_basis.interpolate(V_edges)

_, interpolators_grad = discrete_basis.interpolate(discrete_basis)

h_T = discrete_basis.mesh["cells", "length"]
h_E = discrete_basis.mesh["interior_edges", "length"].unsqueeze(-2)
n_E = discrete_basis.mesh["interior_edges", "normals"].unsqueeze(-2)

# ---------------------- Residual Parameters ----------------------#


def rhs(x, y):
    """Right-hand side function."""
    return 2.0 * math.pi**2 * torch.sin(math.pi * x) * torch.sin(math.pi * y)


def residual(basis, neural_network):
    """Residual of the PDE."""
    x, y = torch.split(basis.integration_points, 1, dim=-1)

    grad = interpolators_grad(neural_network)

    v = basis.v
    v_grad = basis.v_grad
    rhs_value = rhs(x, y)

    return rhs_value * v - (v_grad @ grad.mT)


def gram_matrix(basis):
    """Gram matrix of the basis functions."""
    v_grad = basis.v_grad

    return v_grad @ v_grad.mT


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
    integration_points = basis.integration_points
    x, y = torch.split(integration_points, 1, dim=-1)

    # return triangle_size**2 * (rhs(x, y) + interpolators_grad(nn))

    return triangle_size**2 * (rhs(x, y) + nn(integration_points))


gram_matrix_inverse = torch.inverse(
    discrete_basis.reduce(discrete_basis.integrate_bilinear_form(gram_matrix))
)

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


exact_norm = torch.sqrt(torch.sum(discrete_basis.integrate_functional(h1_exact)))

# ---------------------- Training ----------------------#


def training_step(neural_network):
    """Training step for the neural network."""
    residual_vector = discrete_basis.reduce(
        discrete_basis.integrate_linear_form(residual, neural_network.gradient)
    )

    loss_value = residual_vector.T @ (gram_matrix_inverse @ residual_vector)

    posteriori = (
        discrete_basis.integrate_functional(rhs_term, h_T, neural_network) ** 2
    ).sum() + (V_edges.integrate_functional(jump, n_E, h_E, neural_network) ** 2).sum()

    # loss_value = torch.sum(residual_vector**2) + posteriori

    loss_value += posteriori

    relative_loss = torch.sqrt(loss_value) / exact_norm**2

    h1_error = torch.sqrt(
        torch.sum(
            discrete_basis.integrate_functional(
                h1_norm, neural_network, neural_network.gradient
            )
        )
    )

    return loss_value, relative_loss, h1_error / exact_norm


model = Model(
    neural_network=NN,
    training_step=training_step,
    epochs=8000,
    optimizer=torch.optim.Adam,
    optimizer_kwargs={"lr": 0.001},
    # learning_rate_scheduler=torch.optim.lr_scheduler.ExponentialLR,
    # scheduler_kwargs={"gamma": 0.99**100},
    use_early_stopping=True,
    early_stopping_patience=25,
    min_delta=1e-16,
)

model.train()

# ---------------------- Plotting ----------------------#

model.load_optimal_parameters()

h1_error_plot = (
    torch.sqrt(discrete_basis.integrate_functional(h1_norm, NN, NN.gradient))
    .squeeze(-1)
    .numpy(force=True)
)

figure_solution, axis_solution = plt.subplots()

c4e = torch.Tensor.numpy(discrete_basis.mesh["cells", "coordinates"], force=True)

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
        "title": " RVPINNs + a posteriori estimator",
    }
)

plt.show()
