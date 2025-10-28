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
    FeedForwardNeuralNetwork as NeuralNetwork,
    DistanceFunctionBC,
)

torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
torch.set_default_dtype(torch.float64)


# ---------------------- Neural Network Parameters ----------------------#

# class BoundaryConstrain(torch.nn.Module):
#     """Class to strongly apply bc"""

#     def forward(self, inputs):
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
    nb_hidden_layers=4,
    neurons_per_layers=15,
    # boundary_condition_modifier=BoundaryConstrain(),
    boundary_condition_modifier=DistanceFunctionBC(segments),
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

    return torch.tensor(
        initial_centers, device=torch.get_default_device()
    ), torch.tensor(initial_radius, device=torch.get_default_device()).unsqueeze(-1)


centers, radius = generate_patches_info(5)

patches = Patches(centers, radius)

mesh_data = tr.triangulate(
    {"vertices": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]},
    "Dqena" + str(0.5**10),
)

mesh = MeshTri(triangulation=mesh_data)

elements = ElementTri(polynomial_order=1, integration_order=4)

discrete_basis = PatchesBasis(patches, elements)

error_basis = Basis(mesh, elements)

# ---------------------- Residual Parameters ----------------------#

EXPONENTIAL_COEFFICIENT = 5
SCALING_CONSTANT = 1


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


def residual(
    basis: Basis, nn_grad: torch.Tensor, value_rhs: torch.Tensor
) -> torch.Tensor:
    """Residual of the PDE."""
    return value_rhs * basis.v - (basis.v_grad @ nn_grad.mT)


def gram_matrix(basis: Basis) -> torch.Tensor:
    """Gram matrix of the basis functions."""
    return basis.v_grad @ basis.v_grad.mT


gram_matrix_inverse = torch.inverse(
    discrete_basis.reduce(discrete_basis.integrate_bilinear_form(gram_matrix))
    .unsqueeze(-1)
    .unsqueeze(-1)
)

# ---------------------- Error Parameters ----------------------#


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


def h1_exact(
    _,
    value: torch.Tensor,
    value_dx: torch.Tensor,
    value_dy: torch.Tensor,
) -> torch.Tensor:
    """H1 norm of the exact solution."""
    return value**2 + value_dx**2 + value_dy**2


def h1_norm(
    _,
    solution_value: torch.Tensor,
    solution_grad: torch.Tensor,
    value: torch.Tensor,
    dx: torch.Tensor,
    dy: torch.Tensor,
) -> torch.Tensor:
    """H1 norm of the neural network solution."""
    nn_dx, nn_dy = torch.split(solution_grad, 1, dim=-1)

    return (value - solution_value) ** 2 + (dx - nn_dx) ** 2 + (dy - nn_dy) ** 2


# ---------------------- Training ----------------------#

integration_points = discrete_basis.integration_points
error_integration_points = error_basis.integration_points

# Precompute values

rhs_value = rhs(integration_points)
exact_value = exact(error_integration_points)
exact_dx_value = exact_dx(error_integration_points)
exact_dy_value = exact_dy(error_integration_points)
exact_norm = torch.sqrt(
    torch.sum(
        error_basis.integrate_functional(
            h1_exact, exact_value, exact_dx_value, exact_dy_value
        )
    )
)

values = [
    rhs_value,
    exact_value,
    exact_dx_value,
    exact_dy_value,
    exact_norm,
    gram_matrix_inverse,
]


def training_step(
    neural_network: NeuralNetwork,
    basis_patches: PatchesBasis,
    basis_error: Basis,
    precomputed_values: list,
):
    """Training step for the neural network."""

    (
        value_rhs,
        value_exact,
        value_exact_dx,
        value_exact_dy,
        norm_exact,
        matrix,
        # triangle_size,
        # edge_size,
        # normals_edges,
    ) = precomputed_values

    _, nn_grad = neural_network.value_and_gradient(basis_patches.integration_points)

    residual_vector = basis_patches.reduce(
        basis_patches.integrate_linear_form(residual, nn_grad, value_rhs)
    ).unsqueeze(-1)

    # loss_value = torch.sum(residual_vector**2)

    loss_value = (residual_vector.mT @ (matrix @ residual_vector)).sum()

    nn_value_error, nn_grad_error = neural_network.value_and_gradient(
        basis_error.integration_points
    )

    h1_error = torch.sqrt(
        torch.sum(
            basis_error.integrate_functional(
                h1_norm,
                nn_value_error,
                nn_grad_error,
                value_exact,
                value_exact_dx,
                value_exact_dy,
            )
        )
    )

    relative_loss = torch.sqrt(loss_value) / norm_exact

    return loss_value, relative_loss, h1_error / norm_exact


model = Model(
    neural_network=NN,
    training_step=lambda nn: training_step(nn, discrete_basis, error_basis, values),
    epochs=12000,
    optimizer=torch.optim.Adam,
    optimizer_kwargs={"lr": 1e-4},
    # learning_rate_scheduler=torch.optim.lr_scheduler.ExponentialLR,
    # scheduler_kwargs={"gamma": 0.9999},
    use_early_stopping=True,
    early_stopping_patience=120,
    min_delta=1e-15,
)


model.train()

# ---------------------- Plotting ----------------------#

model.load_optimal_parameters()

opt_nn_value, opt_nn_grad = NN.value_and_gradient(error_basis.integration_points)

h1_error_plot = (
    torch.sqrt(
        error_basis.integrate_functional(
            h1_norm,
            opt_nn_value,
            opt_nn_grad,
            exact_value,
            exact_dx_value,
            exact_dy_value,
        )
    )
    .squeeze(-1)
    .numpy(force=True)
)

figure_solution, axis_solution = plt.subplots()

c4e = torch.Tensor.numpy(error_basis.mesh["cells", "coordinates"], force=True)

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

# model.plot_training_history(
#     plot_names={
#         "loss": r"$\mathcal{L}(u_{\theta})$",
#         "validation": r"$\frac{\sqrt{\mathcal{L}(u_{\theta})}}{\|u\|_U}$",
#         "accuracy": r"$\frac{\|u-u_{\theta}\|_U}{\|u_{\theta}\|_U}$",
#         "title": "Training History",
#     }
# )

loss_history, validation_history, accuracy_history = model.get_training_history()

fig_loss, ax_loss = plt.subplots()
ax_loss.semilogy(
    loss_history,
    label=r"$\mathcal{L}_{r_{h}}(u_{\theta})$",
    linestyle="-",
)
ax_loss.semilogy(
    accuracy_history,
    label=r"$\frac{\|u_{\text{ex}}-u_{\theta}\|_U}{\|u_{\text{ex}}\|_U}$",
    linestyle=":",
)
ax_loss.set_xlabel("# Epochs")
ax_loss.set_ylabel("Value")
ax_loss.set_title("Training History")
ax_loss.legend()

fig_convergence, ax_convergence = plt.subplots()
ax_convergence.semilogy(
    validation_history,
    label=r"$\frac{\sqrt{\mathcal{L}_{r_{h}}(u_{\theta})}}{\|u_{\text{ex}}-u_{\theta}\|_U}$",
    linestyle="--",
)
ax_convergence.set_xlabel("# Epochs")
ax_convergence.set_ylabel("Value")
ax_convergence.set_title("Validation History")
ax_convergence.legend()


# figure_residuals, axis_residuals = plt.subplots()

# axis_residuals.semilogy(residual_history, linestyle="-", label="residual")
# axis_residuals.semilogy(bulk_history, linestyle="--", label="bulk")
# axis_residuals.semilogy(jump_history, linestyle=":", label="jump")

# axis_residuals.set_xlabel("# Epochs")
# axis_residuals.set_ylabel("Value")
# axis_residuals.set_title("Value of components of Loss over training phase")
# axis_residuals.legend()
# figure_residuals.tight_layout()


plt.show()
