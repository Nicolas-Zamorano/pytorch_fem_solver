"# Example of solving a Poisson equation using a neural network and FEM basis functions."

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
    # DistanceFunctionBC,
)

# pyright: reportCallIssue=false

# torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
# torch.cuda.empty_cache()
torch.set_default_dtype(torch.float64)


# ---------------------- Neural Network Parameters ----------------------#

NN = NeuralNetwork(
    input_dimension=2,
    output_dimension=1,
    nb_hidden_layers=2,
    neurons_per_layers=50,
    use_xavier_initialization=True,
)

# ---------------------- FEM Parameters ----------------------#

K_INTERPOLATION = 3
K_TEST_FUNCTIONS = 1
ORDER_PRECISION_INTEGRATION = 2

mesh_data_coarser = tr.triangulate(
    {"vertices": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]},
    "Dqena" + str(0.5**8),
)

mesh_coarser = MeshTri(triangulation=mesh_data_coarser)

elements_coarser = ElementTri(
    polynomial_order=K_INTERPOLATION, integration_order=ORDER_PRECISION_INTEGRATION
)

basis_coarser = Basis(mesh_coarser, elements_coarser)

new_vertices = basis_coarser.coords_4_global_dofs.numpy(force=True)
new_segments = basis_coarser.vertices_4_new_edges.numpy(force=True)

centroids = torch.Tensor.numpy(
    mesh_coarser["cells", "coordinates"].mean(dim=-2), force=True
)
new_regions = [[c[0], c[1], i, 0] for i, c in enumerate(centroids)]

mesh_data_finer = tr.triangulate(
    dict(
        vertices=new_vertices,
        segments=new_segments,
        regions=new_regions,
    ),
    "penA",
)

mesh_finer = MeshTri(triangulation=mesh_data_finer)

elements_finer = ElementTri(
    polynomial_order=K_TEST_FUNCTIONS, integration_order=ORDER_PRECISION_INTEGRATION
)

basis_finer = Basis(mesh_finer, elements_finer)

interpolation_function, grad_interpolation_function = basis_coarser.interpolate(
    basis_finer
)

elements_edges_finer = ElementLine(
    polynomial_order=K_TEST_FUNCTIONS, integration_order=ORDER_PRECISION_INTEGRATION
)

basis_edges_finer = InteriorEdgesBasis(mesh_finer, elements_edges_finer)

_, grad_interpolation_edges_function = basis_finer.interpolate(basis_edges_finer)

basis_finer.mesh.compute_edges_values()

h_T = basis_finer.mesh["cells", "length"]
h_E = basis_finer.mesh["interior_edges", "length"]
n_E = basis_finer.mesh["interior_edges", "normals"].unsqueeze(-2)
boundary_dofs = basis_finer.basis_parameters["boundary_dofs"].squeeze(-1)


# ---------------------- Residual Parameters ----------------------#


def residual(
    basis: Basis, nn_grad: torch.Tensor, value_rhs: torch.Tensor
) -> torch.Tensor:
    """Residual of the PDE."""
    return value_rhs * basis.v - (basis.v_grad @ nn_grad.mT)


# def gram_matrix(basis: Basis) -> torch.Tensor:
#     """Gram matrix of the basis functions."""
#     return basis.v_grad @ basis.v_grad.mT


# gram_matrix_inverse = torch.inverse(
# basis_finer.integrate_bilinear_form(gram_matrix)
# )


def jump(
    _,
    normal_elements: torch.Tensor,
    nn_grad_jump: torch.Tensor,
) -> torch.Tensor:
    """Jump term for discontinuous solutions"""
    nn_grad_plus, nn_grad_minus = torch.unbind(nn_grad_jump, dim=-4)
    return ((nn_grad_plus - nn_grad_minus) * normal_elements).sum(-1, keepdim=True) ** 2


def bulk(
    _,
    laplacian: torch.Tensor,
    value_rhs: torch.Tensor,
) -> torch.Tensor:
    """Residual term for the right-hand side"""
    return (value_rhs + laplacian) ** 2


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

# EXPONENTIAL_COEFFICIENT = 5
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


# ---------------------- Training ----------------------#


integration_points = basis_finer.integration_points

# Precompute values

rhs_value = rhs(integration_points)
exact_value = exact(integration_points)
exact_value_dofs = exact(basis_finer.coords_4_global_dofs)
exact_dx_value = exact_dx(integration_points)
exact_dy_value = exact_dy(integration_points)
exact_norm = torch.sqrt(
    torch.sum(
        basis_finer.integrate_functional(
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
    # gram_matrix_inverse,
    h_T,
    h_E,
    n_E,
    boundary_dofs,
]

bulk_history = []
jump_history = []
residual_history = []


def training_step(
    neural_network: NeuralNetwork,
    coarser_basis: Basis,
    finer_basis: Basis,
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
        # matrix,
        triangle_size,
        edge_size,
        normals_edges,
        dofs_boundary,
    ) = precomputed_values

    nn_value, nn_grad = neural_network.value_and_gradient(
        coarser_basis.coords_4_global_dofs
    )

    nn_value[dofs_boundary] = value_exact_dofs[dofs_boundary]

    nn_jump_grad_interpolated = grad_interpolation_edges_function(nn_value)

    nn_value_interpolated = interpolation_function(nn_value)
    nn_grad_interpolated = grad_interpolation_function(nn_value)
    nn_laplacian_interpolated = grad_interpolation_function(nn_grad)

    residual_vector = finer_basis.integrate_linear_form(
        residual,
        nn_grad_interpolated,
        value_rhs,
    )

    loss_value = torch.sum(residual_vector**2)

    # loss_value = residual_vector.T @ (matrix @ residual_vector)

    bulk_value = (
        triangle_size
        * finer_basis.integrate_functional(
            bulk,
            nn_laplacian_interpolated,
            value_rhs,
        )
    ).sum()

    jump_value = (
        torch.sqrt(edge_size)
        * basis_edges_finer.integrate_functional(
            jump,
            normals_edges,
            nn_jump_grad_interpolated,
        )
    ).sum()

    residual_history.append(loss_value.item())
    bulk_history.append(bulk_value.item())
    jump_history.append(jump_value.item())

    loss_value += bulk_value + jump_value

    nn_dx_interpolated, nn_dy_interpolated = torch.split(
        nn_grad_interpolated, 1, dim=-1
    )

    h1_error = torch.sqrt(
        torch.sum(
            finer_basis.integrate_functional(
                h1_norm,
                value_exact - nn_value_interpolated,
                value_exact_dx - nn_dx_interpolated,
                value_exact_dy - nn_dy_interpolated,
            )
        )
    )

    relative_loss = torch.sqrt(loss_value) / h1_error

    return loss_value, relative_loss, h1_error / norm_exact


model = Model(
    neural_network=NN,
    training_step=lambda nn: training_step(
        nn,
        basis_coarser,
        basis_finer,
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

opt_nn_value = NN(basis_coarser.coords_4_global_dofs)


opt_nn_value[boundary_dofs] = exact_value_dofs[boundary_dofs]

opt_nn_interpolated = interpolation_function(opt_nn_value)
opt_nn_grad = grad_interpolation_function(opt_nn_value)
opt_nn_dx, opt_nn_dy = torch.split(opt_nn_grad, 1, dim=-1)

h1_error_plot = (
    torch.sqrt(
        # basis_coarser.integrate_functional(
        basis_finer.integrate_functional(
            h1_norm,
            exact_value - opt_nn_interpolated,
            exact_dx_value - opt_nn_dx,
            exact_dy_value - opt_nn_dy,
        )
    )
    .squeeze(-1)
    .numpy(force=True)
)

figure_solution, axis_solution = plt.subplots()

coordinates_4_triangles = torch.Tensor.numpy(
    basis_finer.mesh["cells", "coordinates"], force=True
)


triangles_plot = PolyCollection(
    coordinates_4_triangles,  # type: ignore
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
