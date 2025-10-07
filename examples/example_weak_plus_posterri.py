"# Example of solving a Poisson equation using a neural network and FEM basis functions."

import math

import matplotlib.pyplot as plt
import torch
import triangle as tr
from matplotlib.collections import PolyCollection

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
    nb_hidden_layers=4,
    neurons_per_layers=15,
    boundary_condition_modifier=BoundaryConstrain(),
)

# ---------------------- FEM Parameters ----------------------#

mesh_data = tr.triangulate(
    {"vertices": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]},
    "Dqena" + str(0.5**8),
)

mesh = MeshTri(triangulation=mesh_data)

elements = ElementTri(polynomial_order=1, integration_order=4)

discrete_basis = Basis(mesh, elements)

elements_1D = ElementLine(polynomial_order=1, integration_order=2)

V_edges = InteriorEdgesBasis(mesh, elements_1D)

_, interpolator_to_edges_grad = discrete_basis.interpolate(V_edges)

h_T = discrete_basis.mesh["cells", "length"].reshape(-1, 1, 3, 1)
h_E = discrete_basis.mesh["interior_edges", "length"].unsqueeze(-2)
n_E = discrete_basis.mesh["interior_edges", "normals"].unsqueeze(-2)

# ---------------------- Residual Parameters ----------------------#


def rhs(x):
    """Right-hand side function."""
    return (
        2.0
        * math.pi**2
        * torch.sin(math.pi * x[..., [0]])
        * torch.sin(math.pi * x[..., [1]])
    )


def residual(basis, neural_network):
    """Residual of the PDE."""
    grad = neural_network.gradient(basis.integration_points)

    v = basis.v
    v_grad = basis.v_grad
    rhs_value = rhs(basis.integration_points)

    return rhs_value * v - (v_grad @ grad.mT)


def residual_vpinns(
    basis: Basis, neural_network: NeuralNetwork, triangle_size: torch.Tensor
):
    """VPINNs Residual of the PDE."""

    grad = neural_network.gradient(basis.integration_points)
    lap = neural_network.laplacian(basis.integration_points)

    v = basis.v
    v_grad = basis.v_grad
    rhs_value = rhs(basis.integration_points)

    return (
        rhs_value * v - (v_grad @ grad.mT) + triangle_size**2 * (rhs_value + lap) ** 2
    )


def gram_matrix(basis):
    """Gram matrix of the basis functions."""
    v_grad = basis.v_grad

    return v_grad @ v_grad.mT


def jump(_, normal_elements, edge_size, neural_network):
    """Jump term for discontinuous solutions"""
    interpolator_u_grad_plus, interpolator_u_grad_minus = torch.unbind(
        interpolator_to_edges_grad(neural_network), dim=-4
    )
    return (
        edge_size
        * (
            (interpolator_u_grad_plus * normal_elements).sum(-1, keepdim=True)
            + (interpolator_u_grad_minus * -normal_elements).sum(-1, keepdim=True)
        )
        ** 2
    )


def bulk(basis, triangle_size, neural_network):
    """Residual term for the right-hand side"""
    return triangle_size**2 * (
        rhs(basis.integration_points)
        + neural_network.laplacian(basis.integration_points)
    )


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

bulk_history = []
jump_history = []
residual_history = []


def training_step(neural_network):
    """Training step for the neural network."""

    bulk_value = (
        discrete_basis.integrate_functional(bulk, h_T, neural_network) ** 2
    ).sum()

    jump_value = (
        V_edges.integrate_functional(jump, n_E, h_E, neural_network) ** 2
    ).sum()

    residual_value = discrete_basis.reduce(
        discrete_basis.integrate_linear_form(residual, neural_network)
    )

    residual_vector = residual_value

    residual_matvec = residual_vector.T @ (gram_matrix_inverse @ residual_vector)

    loss_value = residual_matvec + bulk_value + jump_value

    # residual_value = discrete_basis.reduce(
    #     (discrete_basis.integrate_linear_form(residual_vpinns, neural_network, h_T))
    # )

    # loss_value = torch.sum(residual_value**2) + jump_value

    relative_loss = torch.sqrt(loss_value) / exact_norm**2

    h1_error = torch.sqrt(
        torch.sum(
            discrete_basis.integrate_functional(
                h1_norm, neural_network, neural_network.gradient
            )
        )
    )

    bulk_history.append(bulk_value.item())
    jump_history.append(jump_value.item())
    residual_history.append(residual_matvec.item())

    return loss_value, relative_loss, h1_error / exact_norm


model = Model(
    neural_network=NN,
    training_step=training_step,
    epochs=10000,
    optimizer=torch.optim.Adam,
    optimizer_kwargs={"lr": 0.001},
    # learning_rate_scheduler=torch.optim.lr_scheduler.ExponentialLR,
    # scheduler_kwargs={"gamma": 0.99**100},
    use_early_stopping=True,
    early_stopping_patience=10,
    min_delta=1e-15,
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
figure_solution.tight_layout()

axis_solution.set_xlabel("x")
axis_solution.set_ylabel("y")
axis_solution.set_xlim((0, 1))
axis_solution.set_ylim((0, 1))
axis_solution.set_title(r"$H^1$ error of solution")
color_bar = plt.colorbar(collection, ax=axis_solution)
color_bar.set_label(r"$H^1$ error")

model.plot_training_history(
    plot_names={
        "loss": r"$\mathcal{L}(u_{\theta})$",
        "validation": r"$\frac{\sqrt{\mathcal{L}(u_{\theta})}}{\|u\|_U}$",
        "accuracy": r"$\frac{\|u-u_{\theta}\|_U}{\|u_{\theta}\|_U}$",
        "title": "RVPINNs + Posteriori Estimator",
    }
)

figure_residuals, axis_residuals = plt.subplots()

axis_residuals.semilogy(residual_history, linestyle="-", label="residual")
axis_residuals.semilogy(bulk_history, linestyle="--", label="bulk")
axis_residuals.semilogy(jump_history, linestyle=":", label="jump")

axis_residuals.set_xlabel("# Epochs")
axis_residuals.set_ylabel("Value")
axis_residuals.set_title("Value of components of Loss over training phase")
axis_residuals.legend()
figure_residuals.tight_layout()

plt.show()
