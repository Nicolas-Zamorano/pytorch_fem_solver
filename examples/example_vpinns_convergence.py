"# Example of solving a Poisson equation using a neural network and FEM basis functions."

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import torch
import triangle as tr
import numpy as np

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

NN_initial_parameters = NN.state_dict().copy()

# ---------------------- FEM Parameters ----------------------#

K_INTERPOLATION = 3
K_TEST_FUNCTIONS = 1
ORDER_PRECISION_INTEGRATION = 2

INITIAL_TRIANGLE_SIZE = 0.5
INITIAL_EXPONENT = 1
NB_REFINEMENTS = 9

elements_coarser = ElementTri(
    polynomial_order=K_INTERPOLATION, integration_order=ORDER_PRECISION_INTEGRATION
)

elements_finer = ElementTri(
    polynomial_order=K_TEST_FUNCTIONS, integration_order=ORDER_PRECISION_INTEGRATION
)

elements_edges_finer = ElementLine(
    polynomial_order=K_TEST_FUNCTIONS, integration_order=ORDER_PRECISION_INTEGRATION
)

mesh_data_coarser = tr.triangulate(
    {"vertices": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]},
    "qa" + str(INITIAL_TRIANGLE_SIZE**INITIAL_EXPONENT),
)

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


def training_step(
    neural_network: NeuralNetwork,
    coarser_basis: Basis,
    finer_basis: Basis,
    basis_edges_finer: InteriorEdgesBasis,
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
        grad_interpolation_edges_function,
        interpolation_function,
        grad_interpolation_function,
    ) = precomputed_values

    nn_value, nn_grad = neural_network.value_and_gradient(
        coarser_basis.coordinates_4_global_dofs
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


# ---------------------- Training ----------------------#

# Convergence study tracking
mesh_sizes = []
h1_errors = []

for i in range(NB_REFINEMENTS):

    if i > 0:
        # Refine existing mesh
        mesh_data_coarser = tr.triangulate(
            mesh_data_coarser,
            "ra" + str(INITIAL_TRIANGLE_SIZE ** (INITIAL_EXPONENT + i + 1)),
        )

    mesh_coarser = MeshTri(triangulation=mesh_data_coarser)

    basis_coarser = Basis(mesh_coarser, elements_coarser)

    new_vertices = basis_coarser.coordinates_4_global_dofs.numpy(force=True)
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

    basis_finer = Basis(mesh_finer, elements_finer)

    interpolation_function, grad_interpolation_function = basis_coarser.interpolate(
        basis_finer
    )

    basis_edges_finer = InteriorEdgesBasis(mesh_finer, elements_edges_finer)

    _, grad_interpolation_edges_function = basis_finer.interpolate(basis_edges_finer)

    basis_finer.mesh.compute_edges_values()

    h_T = basis_finer.mesh["cells", "length"]
    h_E = basis_finer.mesh["interior_edges", "length"]
    n_E = basis_finer.mesh["interior_edges", "normals"].unsqueeze(-2)
    boundary_dofs = basis_finer.basis_parameters["boundary_dofs"].squeeze(-1)

    # Compute characteristic mesh size (maximum triangle diameter)
    mesh_size = torch.max(h_T).item()
    mesh_sizes.append(mesh_size)
    print(f"\nRefinement {i+1}/{NB_REFINEMENTS}: mesh size h = {mesh_size:.6f}")

    integration_points = basis_finer.integration_points

    # Precompute values

    rhs_value = rhs(integration_points)
    exact_value = exact(integration_points)
    exact_value_dofs = exact(basis_finer.coordinates_4_global_dofs)
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
        grad_interpolation_edges_function,
        interpolation_function,
        grad_interpolation_function,
    ]

    bulk_history = []
    jump_history = []
    residual_history = []

    NN.load_state_dict(NN_initial_parameters)

    model = Model(
        neural_network=NN,
        training_step=lambda nn: training_step(
            nn,
            basis_coarser,
            basis_finer,
            basis_edges_finer,
            values,
        ),
        epochs=12000,
        optimizer=torch.optim.Adam,
        optimizer_kwargs={"lr": 1e-3},
        # learning_rate_scheduler=torch.optim.lr_scheduler.ExponentialLR,
        # scheduler_kwargs={"gamma": 0.9999},
        use_early_stopping=True,
        early_stopping_patience=120,
        min_delta=1e-15,
    )

    model.train()

    # ---------------------- Plotting ----------------------#

    model.load_optimal_parameters()

    opt_nn_value = NN(basis_coarser.coordinates_4_global_dofs)

    opt_nn_value[boundary_dofs] = exact_value_dofs[boundary_dofs]

    opt_nn_interpolated = interpolation_function(opt_nn_value)
    opt_nn_grad = grad_interpolation_function(opt_nn_value)
    opt_nn_dx, opt_nn_dy = torch.split(opt_nn_grad, 1, dim=-1)

    # Compute and save H¹ error
    h1_error_value = torch.sqrt(
        torch.sum(
            basis_finer.integrate_functional(
                h1_norm,
                exact_value - opt_nn_interpolated,
                exact_dx_value - opt_nn_dx,
                exact_dy_value - opt_nn_dy,
            )
        )
    ).item()

    h1_errors.append(h1_error_value)
    print(f"H¹ error = {h1_error_value:.6e}")

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
    figure_solution.savefig(
        f"solution_refinement_{i+1}.png", dpi=300, bbox_inches="tight"
    )

    model.plot_training_history(
        plot_names={
            "loss": r"$\mathcal{L}(u_{\theta})$",
            "validation": r"$\frac{\sqrt{\mathcal{L}(u_{\theta})}}{\|u\|_U}$",
            "accuracy": r"$\frac{\|u-u_{\theta}\|_U}{\|u_{\theta}\|_U}$",
            "title": "Training History",
        }
    )
    plt.gcf().savefig(
        f"training_history_refinement_{i+1}.png", dpi=300, bbox_inches="tight"
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
    figure_residuals.savefig(
        f"residuals_refinement_{i+1}.png", dpi=300, bbox_inches="tight"
    )

    plt.close("all")  # Close figures to free memory

# ---------------------- Convergence Plot ----------------------#

# Compute polynomial fit in log-log space
log_mesh_sizes = np.log10(mesh_sizes)
log_h1_errors = np.log10(h1_errors)
slope, intercept = np.polyfit(log_mesh_sizes, log_h1_errors, 1)
fit_line = 10**intercept * np.array(mesh_sizes) ** slope

figure_convergence, axis_convergence = plt.subplots(dpi=300)

axis_convergence.loglog(
    mesh_sizes,
    h1_errors,
    "o",
    linewidth=2,
    markersize=8,
    markeredgecolor="black",
    label=rf"$H^1$ error (decay rate = {slope:.2f})",
)

# Plot the fitted line
axis_convergence.loglog(
    mesh_sizes, fit_line, "--", linewidth=1.5, alpha=0.7, label=f"Fitted slope"
)

# Add reference slopes
# h_ref = np.array(mesh_sizes)
# if len(mesh_sizes) > 1:
#     # O(h) reference line
#     c1 = h1_errors[0] / mesh_sizes[0]
#     axis_convergence.loglog(
#         h_ref, c1 * h_ref, ":", linewidth=1.5, label=r"$O(h)$", alpha=0.5, color="gray"
#     )

#     # O(h²) reference line
#     c2 = h1_errors[0] / (mesh_sizes[0] ** 2)
#     axis_convergence.loglog(
#         h_ref,
#         c2 * h_ref**2,
#         "-.",
#         linewidth=1.5,
#         label=r"$O(h^2)$",
#         alpha=0.5,
#         color="gray",
#     )

axis_convergence.set_xlabel("Mesh size $h$", fontsize=12)
axis_convergence.set_ylabel(r"$H^1$ error", fontsize=12)
axis_convergence.set_title("Convergence Study", fontsize=14)
axis_convergence.legend(fontsize=10)
axis_convergence.grid(True, which="both", alpha=0.3)
figure_convergence.tight_layout()
figure_convergence.savefig("convergence_study.png", dpi=300, bbox_inches="tight")

# Save convergence data to file
np.savetxt(
    "convergence_data.txt",
    np.column_stack([mesh_sizes, h1_errors]),
    header="mesh_size h1_error",
    fmt="%.10e",
)

print("\n" + "=" * 50)
print("CONVERGENCE STUDY RESULTS")
print("=" * 50)
print(f"Decay rate (slope in log-log): {slope:.4f}")
for i, (h, err) in enumerate(zip(mesh_sizes, h1_errors)):
    print(f"Refinement {i+1}: h = {h:.6e}, H¹ error = {err:.6e}")
print("=" * 50)

plt.show()
