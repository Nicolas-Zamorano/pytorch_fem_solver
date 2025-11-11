"Convergence Study for 2D FEM using torch_fem"

import os
from math import pi
import matplotlib.pyplot as plt
import numpy as np
import torch
import triangle as tr
from matplotlib.collections import PolyCollection

from torch_fem import Basis, MeshTri, ElementTri

# torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
# torch.cuda.empty_cache()
torch.set_default_dtype(torch.float64)

# ---------------------- FEM Parameters ----------------------#

INITIAL_VALUE = 0.5
EXPONENT = 6
POLYNOMIAL_ORDER = 3


elements = ElementTri(
    polynomial_order=POLYNOMIAL_ORDER, integration_order=2 * POLYNOMIAL_ORDER
)

# ---------------------- Residual Parameters ----------------------#

EXPONENTIAL_COEFFICIENT = 5
SCALING_CONSTANT = 1


def bilinear_form(basis: Basis) -> torch.Tensor:
    """Bilinear form."""
    return basis.v_grad @ basis.v_grad.mT


def linear_form(basis: Basis, rhs_values: torch.Tensor) -> torch.Tensor:
    """Linear form for the right-hand side."""
    return rhs_values * basis.v


# ---------------------- Error Parameters ----------------------#


# def exact(coordinates: torch.Tensor) -> torch.Tensor:
#     x, y = torch.split(coordinates, 1, -1)
#     return torch.sin(pi * x) * torch.sin(pi * y)


# def exact_dx(coordinates: torch.Tensor) -> torch.Tensor:
#     x, y = torch.split(coordinates, 1, -1)
#     return pi * torch.cos(pi * x) * torch.sin(pi * y)


# def exact_dy(coordinates: torch.Tensor) -> torch.Tensor:
#     x, y = torch.split(coordinates, 1, -1)
#     return pi * torch.sin(pi * x) * torch.cos(pi * y)


# def rhs(coordinates: torch.Tensor) -> torch.Tensor:
#     x, y = torch.split(coordinates, 1, -1)
#     return 2 * pi**2 * torch.sin(pi * x) * torch.sin(pi * y)


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


def h1_norm(
    _,
    value: torch.Tensor,
    value_dx: torch.Tensor,
    value_dy: torch.Tensor,
) -> torch.Tensor:
    """H1 norm of the exact solution."""
    return value**2 + value_dx**2 + value_dy**2


def L2_norm(
    _,
    value: torch.Tensor,
) -> torch.Tensor:
    """L2 norm of the neural network solution."""
    return value**2


# ---------------------- Convergence Study Parameters ----------------------#

CONVERGENCE_LEVELS = 8  # Number of refinement levels
BASE_EXPONENT = EXPONENT  # Starting exponent value
RESULTS_FOLDER = os.path.join(os.getcwd(), "FEM_Convergence_Results")

# Create results directory
os.makedirs(RESULTS_FOLDER, exist_ok=True)

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

    h_T = discrete_basis.mesh["cells", "length"]

    # ---------------------- Recompute precomputed values ----------------------#

    integration_points = discrete_basis.integration_points

    rhs_value = rhs(integration_points)
    exact_value = exact(integration_points)
    exact_dx_value = exact_dx(integration_points)
    exact_dy_value = exact_dy(integration_points)

    b = discrete_basis.integrate_linear_form(linear_form, rhs_value)
    A = discrete_basis.integrate_bilinear_form(bilinear_form)
    solution = discrete_basis.solve(A, b)

    interpolated_solution, interpolated_solution_grad = discrete_basis.interpolate(
        discrete_basis, solution
    )

    interpolated_solution_dx, interpolated_solution_dy = torch.split(
        interpolated_solution_grad, 1, -1
    )

    h1_error = discrete_basis.integrate_functional(
        h1_norm,
        exact_value - interpolated_solution,
        exact_dx_value - interpolated_solution_dx,
        exact_dy_value - interpolated_solution_dy,
    )

    h1_error_plot = torch.sqrt(h1_error.squeeze(-1)).numpy(force=True)

    global_h1_error = torch.sqrt(torch.sum(h1_error)).item()

    # L2_error = discrete_basis.integrate_functional(
    #     L2_norm, exact_value - interpolated_solution
    # )

    # h1_error_plot = torch.sqrt(L2_error.squeeze(-1)).numpy(force=True)

    # global_h1_error = torch.sqrt(torch.sum(L2_error)).item()

    # Store convergence data
    mesh_size = torch.mean(h_T).item()  # Average element size
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
slope, intercept = np.polyfit(log_h, log_error, 1)


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
    "slope": slope,
    "intercept": intercept,
    "levels": CONVERGENCE_LEVELS,
}

torch.save(convergence_data, os.path.join(RESULTS_FOLDER, "convergence_data.pt"))

# Save summary
np.savetxt(
    os.path.join(RESULTS_FOLDER, "convergence_data.csv"),
    np.column_stack((mesh_sizes_np, h1_errors_np, dof_counts_np)),
    header="h, H1_error, DOFs",
    delimiter=",",
)

print(f"\nConvergence study completed! Results saved in '{RESULTS_FOLDER}' folder.")
