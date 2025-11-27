"""Make Convergnace Study for FEM."""

import os
import triangle
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_fem import MeshTri, FemSolver as Solver, TanhProblem as Problem

BASE = 0.5
EXPONENT = 1
NB_REFINEMENTS = 13 - EXPONENT
P_ORDER = 1

torch.set_default_dtype(torch.float64)

RESULTS_FOLDER = os.path.join(
    os.getcwd(),
    "imgs/"
    + Solver.__name__
    + "_Convergence_Results"
    + f"_P{P_ORDER}"
    + "_for_"
    + Problem.__name__,
)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

mesh_sizes = []
h1_errors = []
l2_errors = []
dof_counts = []

for level in range(NB_REFINEMENTS):
    print(f"\n--- Refinement Level {level + 1}/{NB_REFINEMENTS} ---")

    level_folder = os.path.join(RESULTS_FOLDER, f"level_{level+1}")
    os.makedirs(level_folder, exist_ok=True)

    mesh_data = triangle.triangulate(
        {"vertices": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]},
        "Dqena" + str(BASE ** (level + EXPONENT)),
    )

    mesh = MeshTri(triangulation=mesh_data)

    solver = Solver(
        mesh=mesh,
        polynomial_order=P_ORDER,
        integral_order=2 * P_ORDER,
        problem=Problem(),
    )

    solution = solver.solve()

    l2_error, h1_error = solver.compute_error(solution)

    figure_solution, figure_error = solver.plot(solution)

    figure_solution.savefig(os.path.join(level_folder, "solution.png"))
    figure_error.savefig(os.path.join(level_folder, "error.png"))
    plt.close(figure_solution)
    plt.close(figure_error)

    triangle_size = mesh["cells", "length"].max().item()

    mesh_sizes.append(triangle_size)
    l2_errors.append(torch.sqrt(torch.sum(l2_error)).item())
    h1_errors.append(torch.sqrt(torch.sum(h1_error)).item())
    dof_counts.append(solver.basis.coordinates_4_global_dofs.shape[-2])

print("\n--- Convergence Analysis ---")

# Convert to numpy for polynomial fitting
mesh_sizes_np = np.array(mesh_sizes)
h1_errors_np = np.array(h1_errors)
l2_errors_np = np.array(l2_errors)
dof_counts_np = np.array(dof_counts)

# FEM theoretical convergence rates
h1_theoretical_slope = P_ORDER
l2_theoretical_slope = P_ORDER + 1


# Helper to compute fitted line in log-log
def fit_line(x, y):
    log_x = np.log(x)
    log_y = np.log(y)
    slope, intercept = np.polyfit(log_x, log_y, 1)
    return slope, intercept, np.exp(intercept) * x**slope


# Helper to compute theoretical line in log-log
def theory_line(x, y, slope):
    """Generate a theoretical line with the same scaling as the numerical data."""
    C = y[0] / (x[0] ** slope)  # match amplitude at first point
    return C * x**slope


# ================================
#   FIGURE 1: H1 ERROR
# ================================
fig_h1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Numerical fits
h1_slope_h, h1_int_h, h1_fit_h = fit_line(mesh_sizes_np, h1_errors_np)
h1_slope_dof, h1_int_dof, h1_fit_dof = fit_line(dof_counts_np, h1_errors_np)

# Theoretical curves
h1_theory_h = theory_line(mesh_sizes_np, h1_errors_np, h1_theoretical_slope)
h1_theory_dof = theory_line(dof_counts_np, h1_errors_np, -h1_theoretical_slope / 2)

# --- H1 vs mesh size ---
ax1.loglog(
    mesh_sizes_np, h1_fit_h, "-.", color="orange", label=f"fit slope = {h1_slope_h:.2f}"
)
ax1.loglog(
    mesh_sizes_np,
    h1_errors_np,
    "^",
    color="orange",
    markersize=7,
    markeredgecolor="black",
    label="H1 error",
)
# ax1.loglog(mesh_sizes_np, h1_theory_h, "k-.")
ax1.set_xlabel("Mesh size h")
ax1.set_ylabel("H1 error")
ax1.set_title("H1 Error vs Mesh Size")
ax1.legend()
ax1.grid(True)

# --- H1 vs DOFs ---
ax2.loglog(
    dof_counts_np,
    h1_fit_dof,
    "-.",
    color="orange",
    label=f"fit slope = {-h1_slope_dof:.2f}",
)
ax2.loglog(
    dof_counts_np,
    h1_errors_np,
    "^",
    color="orange",
    markersize=7,
    markeredgecolor="black",
    label="H1 error",
)
# ax2.loglog(dof_counts_np, h1_theory_dof, "k-.")
ax2.set_xlabel("Degrees of freedom")
ax2.set_ylabel("H1 error")
ax2.set_title("H1 Error vs DOFs")
ax2.legend()
ax2.grid(True)

fig_h1.tight_layout()
fig_h1.savefig(os.path.join(RESULTS_FOLDER, "convergence_H1.png"), dpi=300)
plt.close(fig_h1)


# ================================
#   FIGURE 2: L2 ERROR
# ================================
fig_l2, (bx1, bx2) = plt.subplots(1, 2, figsize=(12, 5))

# Numerical fits
l2_slope_h, l2_int_h, l2_fit_h = fit_line(mesh_sizes_np, l2_errors_np)
l2_slope_dof, l2_int_dof, l2_fit_dof = fit_line(dof_counts_np, l2_errors_np)

# Theoretical curves
l2_theory_h = theory_line(mesh_sizes_np, l2_errors_np, l2_theoretical_slope)
l2_theory_dof = theory_line(dof_counts_np, l2_errors_np, -l2_theoretical_slope / 2)

# --- L2 vs mesh size ---
bx1.loglog(
    mesh_sizes_np, l2_fit_h, "-.", color="orange", label=f"fit slope = {l2_slope_h:.2f}"
)
bx1.loglog(
    mesh_sizes_np,
    l2_errors_np,
    "^",
    color="orange",
    markersize=7,
    markeredgecolor="black",
    label="L2 error",
)
# bx1.loglog(mesh_sizes_np, l2_theory_h, "k-.")
bx1.set_xlabel("Mesh size h")
bx1.set_ylabel("L2 error")
bx1.set_title("L2 Error vs Mesh Size")
bx1.legend()
bx1.grid(True)

# --- L2 vs DOFs ---
bx2.loglog(
    dof_counts_np,
    l2_fit_dof,
    "-.",
    color="orange",
    label=f"fit slope = {-l2_slope_dof:.2f}",
)
bx2.loglog(
    dof_counts_np,
    l2_errors_np,
    "^",
    color="orange",
    markersize=7,
    markeredgecolor="black",
    label="L2 error",
)
# bx2.loglog(dof_counts_np, l2_theory_dof, "k-.")
bx2.set_xlabel("Degrees of freedom")
bx2.set_ylabel("L2 error")
bx2.set_title("L2 Error vs DOFs")
bx2.legend()
bx2.grid(True)

fig_l2.tight_layout()
fig_l2.savefig(os.path.join(RESULTS_FOLDER, "convergence_L2.png"), dpi=300)
plt.close(fig_l2)
