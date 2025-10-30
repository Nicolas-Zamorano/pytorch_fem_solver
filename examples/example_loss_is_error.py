"""Example of using Example solution to train NN to to solve a Poisson
equation with Dirichlet boundary conditions."""

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import torch
import triangle as tr

from torch_fem import (
    Basis,
    ElementTri,
    MeshTri,
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
        return x * (1 - x) * y * (1 - y)


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
    "Dqea" + str(0.5**8),
)

mesh = MeshTri(triangulation=mesh_data)

elements = ElementTri(polynomial_order=1, integration_order=4)

discrete_basis = Basis(mesh, elements)

# ---------------------- Error Parameters ----------------------#

EXPONENTIAL_COEFFICIENT = 5
SCALING_CONSTANT = 1


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


def h1_norm(
    _,
    value: torch.Tensor,
    value_dx: torch.Tensor,
    value_dy: torch.Tensor,
) -> torch.Tensor:
    """H1 norm of the exact solution."""
    return value**2 + value_dx**2 + value_dy**2


# ---------------------- Training ----------------------#

integration_points = discrete_basis.integration_points

# Precompute values

exact_value = exact(integration_points)
exact_dx_value = exact_dx(integration_points)
exact_dy_value = exact_dy(integration_points)
exact_norm = torch.sqrt(
    torch.sum(
        discrete_basis.integrate_functional(
            h1_norm, exact_value, exact_dx_value, exact_dy_value
        )
    )
)

values = [
    exact_value,
    exact_dx_value,
    exact_dy_value,
    exact_norm,
]

# ---------------------- Training ----------------------#


def training_step(
    neural_network: NeuralNetwork,
    basis: Basis,
    precomputed_values: list,
):
    """Training step for the neural network."""
    (
        value_exact,
        value_exact_dx,
        value_exact_dy,
        norm_exact,
    ) = precomputed_values

    nn_value, nn_grad = neural_network.value_and_gradient(integration_points)

    nn_dx, nn_dy = torch.split(nn_grad, 1, dim=-1)

    loss_value = torch.sum(
        basis.integrate_functional(
            h1_norm,
            value_exact - nn_value,
            value_exact_dx - nn_dx,
            value_exact_dy - nn_dy,
        )
    )

    relative_loss = torch.sqrt(loss_value) / norm_exact**2

    h1_error = torch.sqrt(
        torch.sum(
            basis.integrate_functional(
                h1_norm,
                value_exact - nn_value,
                value_exact_dx - nn_dx,
                value_exact_dy - nn_dy,
            )
        )
    )

    return loss_value, relative_loss, h1_error / exact_norm


model = Model(
    neural_network=NN,
    training_step=lambda nn: training_step(nn, discrete_basis, values),
    epochs=20000,
    optimizer=torch.optim.Adam,
    optimizer_kwargs={"lr": 0.001},
    # learning_rate_scheduler=torch.optim.lr_scheduler.ExponentialLR,
    # scheduler_kwargs={"gamma": 0.99**100},
    use_early_stopping=False,
    early_stopping_patience=5,
    min_delta=1e-16,
)

model.train()

# ---------------------- Plotting ----------------------#

model.load_optimal_parameters()

opt_nn_value, opt_nn_grad = NN.value_and_gradient(discrete_basis.integration_points)
opt_nn_dx, opt_nn_dy = torch.split(opt_nn_grad, 1, dim=-1)

h1_error_plot = (
    torch.sqrt(
        discrete_basis.integrate_functional(
            h1_norm,
            exact_value - opt_nn_value,
            exact_dx_value - opt_nn_dx,
            exact_dy_value - opt_nn_dy,
        )
    )
    .reshape(-1)
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
        "title": "MF-RVPINNs",
    }
)

plt.show()
