"# Example of solving a Poisson equation using a neural network and Patches."

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


centers, radius = generate_patches_info(5)

patches = Patches(centers, radius)

mesh_data = tr.triangulate(
    {"vertices": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]},
    "Dqena" + str(0.5**8),
)

mesh = MeshTri(triangulation=mesh_data)

elements = ElementTri(polynomial_order=1, integration_order=4)

discrete_basis = PatchesBasis(patches, elements)

error_basis = Basis(mesh, elements)

# ---------------------- Residual Parameters ----------------------#


EXPONENTIAL_COEFFICIENT = 2.5
SCALING_CONSTANT = 5


def rhs(coordinates: torch.Tensor) -> torch.Tensor:
    """Right-hand side function."""
    x, y = torch.split(coordinates, 1, -1)

    exponential_value = torch.exp(EXPONENTIAL_COEFFICIENT * x)

    gxx = (
        -2 * (exponential_value - 1)
        + 2 * EXPONENTIAL_COEFFICIENT * (1 - 2 * x) * exponential_value
        + EXPONENTIAL_COEFFICIENT**2 * x * (1 - x) * exponential_value
    )

    fxx = SCALING_CONSTANT * y * (1 - y) * gxx

    fyy = SCALING_CONSTANT * (-2) * x * (1 - x) * (exponential_value - 1)

    lap = fxx + fyy
    return -lap


def exact(coordinates: torch.Tensor) -> torch.Tensor:
    """Exact solution of the PDE."""
    x, y = torch.split(coordinates, 1, -1)
    return (
        SCALING_CONSTANT
        * x
        * y
        * (1 - x)
        * (1 - y)
        * (torch.exp(EXPONENTIAL_COEFFICIENT * x) - 1)
    )


def exact_dx(coordinates: torch.Tensor) -> torch.Tensor:
    """Exact solution derivative with respect to x."""

    x, y = torch.split(coordinates, 1, -1)
    exponential_value = torch.exp(EXPONENTIAL_COEFFICIENT * x)

    return (
        SCALING_CONSTANT
        * y
        * (1 - y)
        * (
            (1 - 2 * x) * (exponential_value - 1)
            + EXPONENTIAL_COEFFICIENT * x * (1 - x) * exponential_value
        )
    )


def exact_dy(coordinates: torch.Tensor) -> torch.Tensor:
    """Exact solution derivative with respect to y."""
    x, y = torch.split(coordinates, 1, -1)
    exponential_value = torch.exp(EXPONENTIAL_COEFFICIENT * x)

    return SCALING_CONSTANT * (1 - 2 * y) * x * (1 - x) * (exponential_value - 1)


def residual(basis, gradient):
    """Residual of the PDE."""

    v = basis.v
    v_grad = basis.v_grad
    rhs_value = rhs(basis.integration_points)

    return rhs_value * v - (v_grad @ gradient.mT)


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
    error_basis.reduce(error_basis.integrate_bilinear_form(gram_matrix))
)


# ---------------------- Error Parameters ----------------------#


def h1_exact(basis):
    """H1 norm of the exact solution."""
    x = basis.integration_points

    return exact(x) ** 2 + exact_dx(x) ** 2 + exact_dy(x) ** 2


def h1_norm(basis, neural_network, gradient):
    """H1 norm of the neural network solution."""
    integration_points = basis.integration_points
    x = integration_points

    nn_dx, nn_dy = torch.split(gradient, 1, dim=-1)

    return (
        (exact(x) - neural_network) ** 2
        + (exact_dx(x) - nn_dx) ** 2
        + (exact_dy(x) - nn_dy) ** 2
    )


exact_norm = torch.sqrt(torch.sum(error_basis.integrate_functional(h1_exact)))

# ---------------------- Training ----------------------#


def training_step(neural_network: FeedForwardNeuralNetwork):
    """Training step for the neural network."""

    _, nn_grad = neural_network.value_and_gradient(discrete_basis.integration_points)

    nn_validation_value, nn_validation_grad = neural_network.value_and_gradient(
        error_basis.integration_points
    )

    residual_vector = discrete_basis.reduce(
        discrete_basis.integrate_linear_form(residual, nn_grad)
    ).unsqueeze(-1)

    loss_value = torch.sum(
        residual_vector.mT @ (gram_matrix_inverse @ residual_vector), dim=0
    )

    # loss_value = torch.sum(residual_vector**2, dim=0)

    validation_residual_vector = error_basis.reduce(
        error_basis.integrate_linear_form(residual, nn_validation_grad)
    )

    validation_loss_value = validation_residual_vector.T @ (
        validation_gram_matrix_inverse @ validation_residual_vector
    )

    validation_loss_value = torch.sqrt(validation_loss_value) / exact_norm**2

    error_h1 = torch.sqrt(
        torch.sum(
            error_basis.integrate_functional(
                h1_norm, nn_validation_value, nn_validation_grad
            )
        )
    )

    return loss_value, validation_loss_value, error_h1 / exact_norm


model = Model(
    neural_network=NN,
    training_step=training_step,
    epochs=8000,
    optimizer=torch.optim.Adam,
    optimizer_kwargs={"lr": 0.001},
    # learning_rate_scheduler=torch.optim.lr_scheduler.ExponentialLR,
    # scheduler_kwargs={"gamma": 0.99**100},
    use_early_stopping=False,
    early_stopping_patience=800,
    min_delta=1e-15,
)


model.train()

# ---------------------- Plotting ----------------------#

model.load_optimal_parameters()

nn_error_value, nn_error_grad = NN.value_and_gradient(error_basis.integration_points)

h1_error = torch.sqrt(
    error_basis.integrate_functional(h1_norm, nn_error_value, nn_error_grad)
).squeeze(-1)

figure_solution, axis_solution = plt.subplots()

c4e = torch.Tensor.numpy(error_basis.mesh["cells", "coordinates"], force=True)

collection = PolyCollection(
    c4e,  # type: ignore
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
