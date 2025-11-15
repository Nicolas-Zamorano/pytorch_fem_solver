"# Example of solving a Poisson equation using a neural network and FEM discrete_basis functions."

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

# pyright: reportCallIssue=false
# pyright: reportArgumentType=false

# torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
# torch.cuda.empty_cache()
torch.set_default_dtype(torch.float64)


# ---------------------- Neural Network Parameters ----------------------#


# class BoundaryConstrain(torch.nn.Module):
#     """Class to strongly apply bc"""

#     def forward(self, inputs: torch.Tensor) -> torch.Tensor:
#         """Boundary condition modifier function."""
#         x, y = torch.split(inputs, 1, dim=-1)
#         return x * (x - 1) * y * (y - 1)


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
    nb_hidden_layers=2,
    neurons_per_layers=50,
    # boundary_condition_modifier=DistanceFunctionBC(segments),
    # boundary_condition_modifier=BoundaryConstrain(),
    use_xavier_initialization=True,
)

# ---------------------- FEM Parameters ----------------------#

mesh_data = tr.triangulate(
    {"vertices": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]},
    "Dqena" + str(0.5**10),
)

mesh = MeshTri(triangulation=mesh_data)

elements = ElementTri(polynomial_order=2, integration_order=6)

discrete_basis = Basis(mesh, elements)

interpolation_function, interpolation_function_grad = discrete_basis.interpolate(
    discrete_basis
)


# ---------------------- Residual Parameters ----------------------#


def residual(
    basis: Basis, nn_grad: torch.Tensor, value_rhs: torch.Tensor
) -> torch.Tensor:
    """Residual of the PDE."""
    return value_rhs * basis.v - (basis.v_grad @ nn_grad.mT)


def h1_norm(
    _,
    value: torch.Tensor,
    value_dx: torch.Tensor,
    value_dy: torch.Tensor,
) -> torch.Tensor:
    """H1 norm"""
    return value**2 + value_dx**2 + value_dy**2


# ---------------------- Right-hand Side and Exact Solution ----------------------#

#### Exponential Case ####

# EXPONENTIAL_COEFFICIENT = 2.5
# SCALING_CONSTANT = 1


# def exact(coordinates: torch.Tensor) -> torch.Tensor:
#     """Exact solution of the PDE."""
#     x, y = torch.split(coordinates, 1, -1)
#     return (
#         SCALING_CONSTANT
#         * x
#         * y
#         * (1 - x)
#         * (1 - y)
#         * (torch.exp(EXPONENTIAL_COEFFICIENT * x) - 1)
#     )


# def exact_dx(coordinates: torch.Tensor) -> torch.Tensor:
#     """Exact solution derivative with respect to x."""

#     x, y = torch.split(coordinates, 1, -1)
#     exponential_value = torch.exp(EXPONENTIAL_COEFFICIENT * x)

#     return (
#         SCALING_CONSTANT
#         * y
#         * (1 - y)
#         * (
#             (1 - 2 * x) * (exponential_value - 1)
#             + EXPONENTIAL_COEFFICIENT * x * (1 - x) * exponential_value
#         )
#     )


# def exact_dy(coordinates: torch.Tensor) -> torch.Tensor:
#     """Exact solution derivative with respect to y."""
#     x, y = torch.split(coordinates, 1, -1)
#     exponential_value = torch.exp(EXPONENTIAL_COEFFICIENT * x)

#     return SCALING_CONSTANT * (1 - 2 * y) * x * (1 - x) * (exponential_value - 1)


# def rhs(coordinates: torch.Tensor) -> torch.Tensor:
#     """Right-hand side function."""
#     x, y = torch.split(coordinates, 1, -1)

#     exponential_value = torch.exp(EXPONENTIAL_COEFFICIENT * x)

#     exact_dxx = (
#         SCALING_CONSTANT
#         * y
#         * (1 - y)
#         * (
#             -2 * (exponential_value - 1)
#             + 2 * EXPONENTIAL_COEFFICIENT * (1 - 2 * x) * exponential_value
#             + EXPONENTIAL_COEFFICIENT**2 * x * (1 - x) * exponential_value
#         )
#     )

#     exact_dyy = SCALING_CONSTANT * (-2) * x * (1 - x) * (exponential_value - 1)

#     lap = exact_dxx + exact_dyy
#     return -lap


#### Tanh Case ####


def exact(coordinates: torch.Tensor) -> torch.Tensor:
    """Exact solution of the PDE."""
    x, y = torch.split(coordinates, 1, -1)
    return torch.tanh(2 * (x**3 - y**4))


def exact_dx(coordinates: torch.Tensor) -> torch.Tensor:
    """Exact solution derivative with respect to x."""

    x, y = torch.split(coordinates, 1, -1)
    return 6 * x**2 * (1.0 / torch.cosh(2 * (x**3 - y**4)) ** 2)


def exact_dy(coordinates: torch.Tensor) -> torch.Tensor:
    """Exact solution derivative with respect to y."""
    x, y = torch.split(coordinates, 1, -1)

    return -8 * y**3 * (1.0 / torch.cosh(2 * (x**3 - y**4)) ** 2)


def rhs(coordinates: torch.Tensor) -> torch.Tensor:
    """Right-hand side function."""
    x, y = torch.split(coordinates, 1, -1)

    return (
        4
        * (1.0 / torch.cosh(2 * (x**3 - y**4)) ** 2)
        * (
            -3 * x
            + 6 * y**2
            + 2 * (9 * x**4 + 16 * y**6) * torch.tanh(2 * (x**3 - y**4))
        )
    )


# ---------------------- Training ----------------------#


integration_points = discrete_basis.integration_points

# Precompute values

rhs_value = rhs(integration_points)
exact_value = exact(integration_points)

boundary_dofs = discrete_basis.basis_parameters["boundary_dofs"].squeeze(-1)

rhs_value = rhs(integration_points)
exact_value = exact(integration_points)
exact_value_dofs = exact(discrete_basis.coords_4_global_dofs)
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
    rhs_value,
    exact_value,
    exact_value_dofs,
    exact_dx_value,
    exact_dy_value,
    exact_norm,
]

bulk_history = []
jump_history = []
residual_history = []


def training_step(
    neural_network: NeuralNetwork,
    basis: Basis,
    precomputed_values: list,
):
    """Training step for the neural network."""

    (
        value_rhs,
        value_exact,
        value_exact_dofs,
        value_exact_dx,
        value_exact_dy,
        norm_exact,
    ) = precomputed_values

    nn_value = neural_network(basis.coords_4_global_dofs)

    nn_value[boundary_dofs] = value_exact_dofs[boundary_dofs]

    nn_interpolated = interpolation_function(nn_value)

    nn_interpolated_grad = interpolation_function_grad(nn_value)

    residual_vector = basis.integrate_linear_form(
        residual,
        nn_interpolated_grad,
        value_rhs,
    )

    loss_value = torch.sum(residual_vector**2)

    # loss_value = residual_vector.T @ (matrix @ residual_vector)

    nn_dx, nn_dy = torch.split(nn_interpolated_grad, 1, dim=-1)

    h1_error = torch.sqrt(
        torch.sum(
            basis.integrate_functional(
                h1_norm,
                value_exact - nn_interpolated,
                value_exact_dx - nn_dx,
                value_exact_dy - nn_dy,
            )
        )
    )

    relative_loss = torch.sqrt(loss_value) / h1_error

    return loss_value, relative_loss, h1_error / norm_exact


model = Model(
    neural_network=NN,
    training_step=lambda nn: training_step(
        nn,
        discrete_basis,
        values,
    ),
    epochs=12000,
    optimizer=torch.optim.Adam,
    optimizer_kwargs={"lr": 1e-3},
    # learning_rate_scheduler=torch.optim.lr_scheduler.ExponentialLR,
    # scheduler_kwargs={"gamma": 0.9999},
    use_early_stopping=True,
    early_stopping_patience=600,
    min_delta=1e-15,
)


model.train()

# ---------------------- Plotting ----------------------#

model.load_optimal_parameters()

opt_nn_value_dofs = NN(discrete_basis.coords_4_global_dofs)

opt_nn_value_dofs[boundary_dofs] = exact_value_dofs[boundary_dofs]

opt_nn_value = interpolation_function(opt_nn_value_dofs)
opt_nn_grad = interpolation_function_grad(opt_nn_value_dofs)

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
    .squeeze(-1)
    .numpy(force=True)
)

figure_solution, axis_solution = plt.subplots()

c4e = torch.Tensor.numpy(discrete_basis.mesh["cells", "coordinates"], force=True)


triangles_plot = PolyCollection(
    c4e,  # type: ignore
    array=h1_error_plot,
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

model.plot_training_history(
    plot_names={
        "loss": r"$\mathcal{L}(u_{\theta})$",
        "validation": r"$\frac{\sqrt{\mathcal{L}(u_{\theta})}}{\|u\|_U}$",
        "accuracy": r"$\frac{\|u-u_{\theta}\|_U}{\|u_{\theta}\|_U}$",
        "title": "Training History",
    }
)

# loss_history, validation_history, accuracy_history = model.get_training_history()

# fig_loss, ax_loss = plt.subplots()
# ax_loss.semilogy(
#     loss_history,
#     label=r"$\mathcal{L}_{r_{h}}(u_{\theta})$",
#     linestyle="-",
# )
# ax_loss.semilogy(
#     accuracy_history,
#     label=r"$\frac{\|u_{\text{ex}}-u_{\theta}\|_U}{\|u_{\text{ex}}\|_U}$",
#     linestyle=":",
# )
# ax_loss.set_xlabel("# Epochs")
# ax_loss.set_ylabel("Value")
# ax_loss.set_title("Training History")
# ax_loss.legend()

# fig_convergence, ax_convergence = plt.subplots()
# ax_convergence.semilogy(
#     validation_history,
#     label=r"$\frac{\sqrt{\mathcal{L}_{r_{h}}(u_{\theta})}}{\|u_{\text{ex}}-u_{\theta}\|_U}$",
#     linestyle="--",
# )
# ax_convergence.set_xlabel("# Epochs")
# ax_convergence.set_ylabel("Value")
# ax_convergence.set_title("Validation History")
# ax_convergence.legend()

plt.show()
