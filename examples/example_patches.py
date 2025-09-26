"# Example of solving a Poisson equation using a neural network and Patches."

import math

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import torch
import triangle as tr

from torch_fem import (
    MeshTri,
    Patches,
    ElementTri,
    PatchesBasis,
    Basis,
    Model,
    FeedForwardNeuralNetwork,
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


NN = FeedForwardNeuralNetwork(
    input_dimension=2,
    output_dimension=1,
    nb_hidden_layers=4,
    neurons_per_layers=15,
    boundary_condition_modifier=BoundaryConstrain(),
    use_xavier_initialization=True,
)

# ---------------------- FEM Parameters ----------------------#


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

patches = Patches(centers, radius)

mesh_data = tr.triangulate(
    {"vertices": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]},
    "Dqena" + str(0.5**8),
)

mesh = MeshTri(triangulation=mesh_data)

elements = ElementTri(polynomial_order=1, integration_order=2)

validation_elements = ElementTri(polynomial_order=1, integration_order=4)

discrete_basis = PatchesBasis(patches, elements)

validation_basis = PatchesBasis(patches, validation_elements)

error_basis = Basis(mesh, elements)

# ---------------------- Residual Parameters ----------------------#


def rhs(x, y):
    """Right-hand side function."""
    return 2.0 * math.pi**2 * torch.sin(math.pi * x) * torch.sin(math.pi * y)


def residual(basis, gradient):
    """Residual of the PDE."""
    integration_points = basis.integration_points
    x, y = torch.split(integration_points, 1, dim=-1)

    grad = gradient(integration_points)

    v = basis.v
    v_grad = basis.v_grad
    rhs_value = rhs(x, y)

    return rhs_value * v - (v_grad @ grad.mT)


def gram_matrix(basis):
    """Gram matrix of the basis functions."""
    v_grad = basis.v_grad

    return v_grad @ v_grad.mT


gram_matrix_inverse = torch.inverse(
    discrete_basis.reduce(discrete_basis.integrate_bilinear_form(gram_matrix))
    .unsqueeze(-1)
    .unsqueeze(-1)
)

validation_gram_matrix_inverse = torch.inverse(
    discrete_basis.reduce(validation_basis.integrate_bilinear_form(gram_matrix))
    .unsqueeze(-1)
    .unsqueeze(-1)
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


exact_norm = torch.sqrt(torch.sum(error_basis.integrate_functional(h1_exact)))

# ---------------------- Training ----------------------#


def training_step(neural_network):
    """Training step for the neural network."""
    residual_vector = discrete_basis.reduce(
        discrete_basis.integrate_linear_form(residual, neural_network.gradient)
    ).unsqueeze(-1)

    loss_value = torch.sum(
        residual_vector.mT @ (gram_matrix_inverse @ residual_vector), dim=0
    )

    # loss_value = torch.sum(residual_vector**2, dim=0)

    validation_residual_vector = validation_basis.reduce(
        discrete_basis.integrate_linear_form(residual, neural_network.gradient)
    ).unsqueeze(-1)

    validation_loss_value = torch.sum(
        validation_residual_vector.mT
        @ (validation_gram_matrix_inverse @ validation_residual_vector),
        dim=0,
    )

    validation_loss_value = torch.sqrt(validation_loss_value) / exact_norm**2

    h1_error = torch.sqrt(
        torch.sum(
            error_basis.integrate_functional(
                h1_norm, neural_network, neural_network.gradient
            )
        )
    )

    return loss_value, validation_loss_value, h1_error / exact_norm


model = Model(
    neural_network=NN,
    training_step=training_step,
    epochs=8000,
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

h1_error = torch.sqrt(
    error_basis.integrate_functional(
        h1_norm, model._neural_network, model._neural_network.gradient
    )
).squeeze(-1)

figure_solution, axis_solution = plt.subplots()

c4e = error_basis.mesh["cells", "coordinates"].numpy(force=True)

collection = PolyCollection(
    c4e,
    array=h1_error.numpy(force=True),
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
        "title": "MF-RVPINNs",
    }
)


plt.show()
