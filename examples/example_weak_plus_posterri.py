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

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Boundary condition modifier function."""
        x, y = torch.split(inputs, 1, dim=-1)
        return x * (x - 1) * y * (y - 1)


NN = NeuralNetwork(
    input_dimension=2,
    output_dimension=1,
    nb_hidden_layers=4,
    neurons_per_layers=15,
    boundary_condition_modifier=BoundaryConstrain(),
    use_xavier_initialization=True,
)

# ---------------------- FEM Parameters ----------------------#

mesh_data = tr.triangulate(
    {"vertices": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]},
    "Dqena" + str(0.5**8),
)

mesh = MeshTri(triangulation=mesh_data)

elements = ElementTri(polynomial_order=1, integration_order=4)

discrete_basis = Basis(mesh, elements)

elements_1D = ElementLine(polynomial_order=1, integration_order=4)

V_edges = InteriorEdgesBasis(mesh, elements_1D)

jump_integration_points = discrete_basis.compute_jump_integration_points(V_edges)

h_T = discrete_basis.mesh["cells", "length"].reshape(-1, 1, 3, 1)
h_E = discrete_basis.mesh["interior_edges", "length"].squeeze(-1)
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


def residual(
    basis: Basis, nn_grad: torch.Tensor, value_rhs: torch.Tensor
) -> torch.Tensor:
    """Residual of the PDE."""
    return value_rhs * basis.v - (basis.v_grad @ nn_grad.mT)


def gram_matrix(basis: Basis) -> torch.Tensor:
    """Gram matrix of the basis functions."""
    return basis.v_grad @ basis.v_grad.mT


def jump(
    _,
    normal_elements: torch.Tensor,
    nn_grad_jump: torch.Tensor,
) -> torch.Tensor:
    """Jump term for discontinuous solutions"""
    interpolator_u_grad_plus, interpolator_u_grad_minus = torch.unbind(
        nn_grad_jump, dim=-4
    )
    return (
        (interpolator_u_grad_plus * normal_elements).sum(-1, keepdim=True)
        + (interpolator_u_grad_minus * -normal_elements).sum(-1, keepdim=True)
    ) ** 2


def bulk(
    _,
    triangle_size: torch.Tensor,
    laplacian: torch.Tensor,
    value_rhs: torch.Tensor,
) -> torch.Tensor:
    """Residual term for the right-hand side"""
    return triangle_size**2 * (value_rhs + laplacian) ** 2


gram_matrix_inverse = torch.inverse(
    discrete_basis.reduce(discrete_basis.integrate_bilinear_form(gram_matrix))
)


# ---------------------- Error Parameters ----------------------#


def exact(x):
    """Exact solution of the PDE."""
    return torch.sin(math.pi * x[..., [0]]) * torch.sin(math.pi * x[..., [1]])


def exact_dx(x):
    """Exact solution derivative with respect to x."""
    return math.pi * torch.cos(math.pi * x[..., [0]]) * torch.sin(math.pi * x[..., [1]])


def exact_dy(x):
    """Exact solution derivative with respect to y."""
    return math.pi * torch.sin(math.pi * x[..., [0]]) * torch.cos(math.pi * x[..., [1]])


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

# Precompute values

rhs_value = rhs(integration_points)
exact_value = exact(integration_points)
exact_dx_value = exact_dx(integration_points)
exact_dy_value = exact_dy(integration_points)
exact_norm = torch.sqrt(
    torch.sum(
        discrete_basis.integrate_functional(
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
    h_T,
    h_E,
    n_E,
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
        value_exact_dx,
        value_exact_dy,
        norm_exact,
        matrix,
        triangle_size,
        edge_size,
        normals_edges,
    ) = precomputed_values

    # nn_value, nn_grad = neural_network.value_and_gradient(basis.integration_points)

    nn_value, nn_grad, nn_laplacian = neural_network.value_and_laplacian(
        basis.integration_points
    )

    _, nn_jump_grad = neural_network.value_and_gradient(jump_integration_points)

    residual_vector = basis.reduce(
        basis.integrate_linear_form(residual, nn_grad, value_rhs)
    )

    # loss_value = torch.sum(residual_vector**2)

    loss_value = residual_vector.T @ (matrix @ residual_vector)

    bulk_value = (
        basis.integrate_functional(bulk, triangle_size, nn_laplacian, value_rhs)
    ).sum()

    jump_value = (
        edge_size * V_edges.integrate_functional(jump, normals_edges, nn_jump_grad)
    ).sum()

    residual_history.append(loss_value.item())
    bulk_history.append(bulk_value.item())
    jump_history.append(jump_value.item())

    loss_value += bulk_value + jump_value

    relative_loss = torch.sqrt(loss_value) / norm_exact

    h1_error = torch.sqrt(
        torch.sum(
            basis.integrate_functional(
                h1_norm, nn_value, nn_grad, value_exact, value_exact_dx, value_exact_dy
            )
        )
    )

    return loss_value, relative_loss, h1_error / norm_exact


model = Model(
    neural_network=NN,
    training_step=lambda nn: training_step(nn, discrete_basis, values),
    epochs=20000,
    optimizer=torch.optim.Adam,
    optimizer_kwargs={"lr": 0.005},
    learning_rate_scheduler=torch.optim.lr_scheduler.ExponentialLR,
    scheduler_kwargs={"gamma": 0.99**200},
    use_early_stopping=True,
    early_stopping_patience=100,
    min_delta=1e-16,
)


model.train()

# ---------------------- Plotting ----------------------#

model.load_optimal_parameters()

opt_nn_value, opt_nn_grad = NN.value_and_gradient(discrete_basis.integration_points)

h1_error_plot = (
    torch.sqrt(
        discrete_basis.integrate_functional(
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
        "title": "RVPINNs exponential",
    }
)

figure_residuals, axis_residuals = plt.subplots()

axis_residuals.semilogy(residual_history, linestyle="-", label="residual")
axis_residuals.semilogy(bulk_history, linestyle="--", label="bulk")
axis_residuals.semilogy(jump_history, linestyle=":", label="jump")

axis_residuals.set_xlabel("# Epochs")
axis_residuals.set_ylabel("Value")
axis_residuals.set_ylim((1e-4, 1e3))
axis_residuals.set_title("Value of components of Loss over training phase")
axis_residuals.legend()
figure_residuals.tight_layout()


plt.show()
