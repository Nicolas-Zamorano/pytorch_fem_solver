"# Example of solving a Poisson equation using a neural network and FEM basis functions."

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import triangle as tr
from matplotlib.collections import PolyCollection

from torch_fem import Basis, DistanceFunctionBC, ElementLine, ElementTri
from torch_fem import FeedForwardNeuralNetwork as NeuralNetwork
from torch_fem import InteriorEdgesBasis, MeshTri, Model

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
    nb_hidden_layers=4,
    neurons_per_layers=15,
    boundary_condition_modifier=DistanceFunctionBC(segments),
    # boundary_condition_modifier=BoundaryConstrain(),
    use_xavier_initialization=True,
)

# ---------------------- FEM Parameters ----------------------#

INITIAL_VALUE = 0.5
EXPONENT = 1

elements = ElementTri(polynomial_order=1, integration_order=4)

elements_1D = ElementLine(polynomial_order=1, integration_order=4)

# ---------------------- Residual Parameters ----------------------#

EXPONENTIAL_COEFFICIENT = 5
SCALING_CONSTANT = 1


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
    nn_grad_plus, nn_grad_minus = torch.unbind(nn_grad_jump, dim=-4)
    return ((nn_grad_plus - nn_grad_minus) * normal_elements).sum(-1, keepdim=True) ** 2


def bulk(
    _,
    laplacian: torch.Tensor,
    value_rhs: torch.Tensor,
) -> torch.Tensor:
    """Residual term for the right-hand side"""
    return (value_rhs + laplacian) ** 2


# ---------------------- Error Parameters ----------------------#


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
        triangle_size * basis.integrate_functional(bulk, nn_laplacian, value_rhs)
    ).sum()

    jump_value = (
        torch.sqrt(edge_size)
        * V_edges.integrate_functional(jump, normals_edges, nn_jump_grad)
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


# ---------------------- Convergence Study Parameters ----------------------#

CONVERGENCE_LEVELS = 12  # Number of refinement levels
BASE_EXPONENT = EXPONENT  # Starting exponent value
RESULTS_FOLDER = "RVPINNs"

# Create results directory
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Store initial NN parameters
initial_nn_state = NN.state_dict().copy()

# Storage for convergence data
mesh_sizes = []
h1_errors = []
dof_counts = []

print("Starting convergence study...")

for level in range(CONVERGENCE_LEVELS):
    print(f"\n--- Refinement Level {level + 1}/{CONVERGENCE_LEVELS} ---")

    # Current mesh parameter
    current_exponent = BASE_EXPONENT + level
    current_mesh_param = INITIAL_VALUE**current_exponent

    print(f"Mesh parameter: {current_mesh_param}")

    # Create level-specific folder
    level_folder = os.path.join(RESULTS_FOLDER, f"level_{level+1}")
    os.makedirs(level_folder, exist_ok=True)

    # ---------------------- Generate new mesh ----------------------#

    mesh_data = tr.triangulate(
        {"vertices": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]},
        "Dqena" + str(current_mesh_param),
    )

    mesh = MeshTri(triangulation=mesh_data)
    discrete_basis = Basis(mesh, elements)
    V_edges = InteriorEdgesBasis(mesh, elements_1D)
    jump_integration_points = V_edges.compute_jump_integration_points(delta=1e-4)

    # Update mesh-dependent quantities
    h_T = discrete_basis.mesh["cells", "length"]
    h_E = discrete_basis.mesh["interior_edges", "length"]
    n_E = discrete_basis.mesh["interior_edges", "normals"].unsqueeze(-2)

    # ---------------------- Recompute precomputed values ----------------------#

    integration_points = discrete_basis.integration_points

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

    gram_matrix_inverse = torch.inverse(
        discrete_basis.reduce(discrete_basis.integrate_bilinear_form(gram_matrix))
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

    # ---------------------- Reset NN and train ----------------------#

    # Reset neural network to initial parameters
    NN.load_state_dict(initial_nn_state)

    # Reset history lists
    bulk_history = []
    jump_history = []
    residual_history = []

    # Create new training step with updated basis and values
    def training_step_current(neural_network):
        """Training step for the current refinement level."""
        return training_step(neural_network, discrete_basis, values)

    model = Model(
        neural_network=NN,
        training_step=training_step_current,
        epochs=10000,
        optimizer=torch.optim.Adam,
        optimizer_kwargs={"lr": 1e-3},
        use_early_stopping=True,
        early_stopping_patience=100,
        min_delta=1e-15,
    )

    model.train()

    # ---------------------- Compute errors and save results ----------------------#

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

    # Compute global H1 error
    global_h1_error = torch.sqrt(
        torch.sum(
            discrete_basis.integrate_functional(
                h1_norm,
                opt_nn_value,
                opt_nn_grad,
                exact_value,
                exact_dx_value,
                exact_dy_value,
            )
        )
    ).item()

    # Store convergence data
    mesh_size = torch.sqrt(torch.mean(h_T)).item()  # Average element size
    dof_count = len(discrete_basis.mesh["vertices", "coordinates"])

    mesh_sizes.append(mesh_size)
    h1_errors.append(global_h1_error)
    dof_counts.append(dof_count)

    print(f"Mesh size: {mesh_size:.6f}")
    print(f"DOFs: {dof_count}")
    print(f"H1 error: {global_h1_error:.6e}")

    # ---------------------- Save plots and data ----------------------#

    # Solution plot
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
    axis_solution.set_title(f"Level {level+1}: Mesh param = {current_mesh_param:.2e}")
    figure_solution.tight_layout()

    plt.savefig(
        os.path.join(level_folder, f"h1_error_plot_level_{level+1}.png"), dpi=300
    )
    plt.close(figure_solution)

    # Training history plot
    model.plot_training_history(
        plot_names={
            "loss": r"$\mathcal{L}(u_{\theta})$",
            "validation": r"$\frac{\sqrt{\mathcal{L}(u_{\theta})}}{\|u\|_U}$",
            "accuracy": r"$\frac{\|u-u_{\theta}\|_U}{\|u_{\theta}\|_U}$",
            "title": f"Training History - Level {level+1}",
        }
    )
    plt.savefig(
        os.path.join(level_folder, f"training_history_level_{level+1}.png"), dpi=300
    )
    plt.close()

    # Residuals plot
    figure_residuals, axis_residuals = plt.subplots()
    axis_residuals.semilogy(residual_history, linestyle="-", label="residual")
    axis_residuals.semilogy(bulk_history, linestyle="--", label="bulk")
    axis_residuals.semilogy(jump_history, linestyle=":", label="jump")
    axis_residuals.set_xlabel("# Epochs")
    axis_residuals.set_ylabel("Value")
    axis_residuals.set_title(f"Loss Components - Level {level+1}")
    axis_residuals.legend()
    figure_residuals.tight_layout()

    plt.savefig(os.path.join(level_folder, f"residuals_level_{level+1}.png"), dpi=300)
    plt.close(figure_residuals)

    # Save data
    torch.save(
        h1_error_plot, os.path.join(level_folder, f"h1_error_data_level_{level+1}.pt")
    )

    # Save level info
    level_info = {
        "level": level + 1,
        "mesh_parameter": current_mesh_param,
        "mesh_size": mesh_size,
        "dof_count": dof_count,
        "h1_error": global_h1_error,
        "exponent": current_exponent,
    }
    torch.save(level_info, os.path.join(level_folder, f"level_info_{level+1}.pt"))

# ---------------------- Convergence Analysis ----------------------#

print("\n--- Convergence Analysis ---")

# Convert to numpy for polynomial fitting
mesh_sizes_np = np.array(mesh_sizes)
h1_errors_np = np.array(h1_errors)
dof_counts_np = np.array(dof_counts)

# Fit polynomial (degree 1) to log-log data for convergence rate
log_h = np.log(mesh_sizes_np)
log_error = np.log(h1_errors_np)

# Fit: log(error) = slope * log(h) + intercept
poly_coeffs = np.polyfit(log_h, log_error, 1)
slope = poly_coeffs[0]
intercept = poly_coeffs[1]

print(f"Convergence rate (slope): {slope:.4f}")
print("Expected theoretical rate for linear elements: ~2.0")

# Create convergence plots
figure_conv, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Error vs mesh size
ax1.loglog(mesh_sizes_np, h1_errors_np, "bo-", label="Computed errors")
ax1.loglog(
    mesh_sizes_np,
    np.exp(intercept) * mesh_sizes_np**slope,
    "r--",
    label=f"Fitted line (slope={slope:.2f})",
)
ax1.set_xlabel("Mesh size h")
ax1.set_ylabel("H1 error")
ax1.set_title("Convergence Study")
ax1.legend()
ax1.grid(True)

# Plot 2: Error vs DOFs
ax2.loglog(dof_counts_np, h1_errors_np, "go-", label="H1 error vs DOFs")
ax2.set_xlabel("Number of DOFs")
ax2.set_ylabel("H1 error")
ax2.set_title("Error vs DOFs")
ax2.legend()
ax2.grid(True)

figure_conv.tight_layout()
plt.savefig(os.path.join(RESULTS_FOLDER, "convergence_study.png"), dpi=300)
plt.show()

# Save convergence data
convergence_data = {
    "mesh_sizes": mesh_sizes_np,
    "h1_errors": h1_errors_np,
    "dof_counts": dof_counts_np,
    "convergence_rate": slope,
    "poly_coeffs": poly_coeffs,
    "levels": CONVERGENCE_LEVELS,
}

torch.save(convergence_data, os.path.join(RESULTS_FOLDER, "convergence_data.pt"))

# Save summary
with open(
    os.path.join(RESULTS_FOLDER, "convergence_summary.txt"), "w", encoding="utf-8"
) as f:
    f.write("Convergence Study Summary\n")
    f.write("=" * 30 + "\n\n")
    f.write(f"Number of refinement levels: {CONVERGENCE_LEVELS}\n")
    f.write(f"Base exponent: {BASE_EXPONENT}\n")
    f.write(f"Convergence rate: {slope:.4f}\n\n")

    f.write("Level Details:\n")
    for i in range(CONVERGENCE_LEVELS):
        f.write(
            f"Level {i+1}: h={mesh_sizes[i]:.6f}, DOFs={dof_counts[i]}, Error={h1_errors[i]:.6e}\n"
        )

print(f"\nConvergence study completed! Results saved in '{RESULTS_FOLDER}' folder.")
