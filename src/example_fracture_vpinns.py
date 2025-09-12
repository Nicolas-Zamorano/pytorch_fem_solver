"""Example of Variational Physics-Informed Neural Networks (VPINNs) for a 3D fracture problem."""

import os
from datetime import datetime

import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import tensordict as td
import torch
import triangle as tr
from matplotlib.collections import PolyCollection

from fem import (
    ElementLine,
    ElementTri,
    FractureBasis,
    FracturesTri,
    InteriorEdgesFractureBasis,
)

from model import FeedForwardNeuralNetwork as NeuralNetwork, Model

torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
torch.set_default_dtype(torch.float64)


def boundary_constraint(x):
    """enforces Dirichlet boundary condition"""
    return (
        (x[..., [0]] + 1)
        * (x[..., [0]] - 1)
        * (x[..., [1]])
        * (x[..., [1]] - 1)
        * (x[..., [2]] + 1)
        * (x[..., [2]] - 1)
    )


NN = NeuralNetwork(
    input_dimension=2,
    output_dimension=1,
    nb_hidden_layers=4,
    neurons_per_layers=25,
    boundary_condition_modifier=boundary_constraint,
)


# ---------------------- FEM Parameters ----------------------#

fracture_2d_data = {
    "vertices": [
        [-1.0, 0.0],
        [1.0, 0.0],
        [-1.0, 1.0],
        [1.0, 1.0],
        [0.0, 0.0],
        [0.0, 1.0],
    ],
    "segments": [[0, 2], [0, 4], [1, 3], [1, 4], [2, 5], [3, 5], [4, 5]],
}

fracture_triangulation = tr.triangulate(fracture_2d_data, "pqsea" + str(0.5**10))

fractures_triangulation = [fracture_triangulation, fracture_triangulation]

fractures_data = torch.tensor(
    [
        [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        [[0.0, 0.0, -1.0], [0.0, 0.0, 1.0], [0.0, 1.0, -1.0], [0.0, 1.0, 1.0]],
    ]
)

mesh = FracturesTri(
    triangulations=fractures_triangulation, fractures_3d_data=fractures_data
)

elements = ElementTri(polynomial_order=1, integration_order=2)

discrete_basis = FractureBasis(mesh, elements)

# ---------------------- Residual Parameters ----------------------#


def rhs(x, y, z):
    """Right-hand side function for facture problem."""
    x_fracture_1, _ = torch.split(x, 1, dim=0)
    y_fracture_1, y_fracture_2 = torch.split(y, 1, dim=0)
    _, z_fracture_2 = torch.split(z, 1, dim=0)

    rhs_fracture_1 = 6.0 * (y_fracture_1 - y_fracture_1**2) * torch.abs(
        x_fracture_1
    ) - 2.0 * (torch.abs(x_fracture_1) ** 3 - torch.abs(x_fracture_1))
    rhs_fracture_2 = -6.0 * (y_fracture_2 - y_fracture_2**2) * torch.abs(
        z_fracture_2
    ) + 2.0 * (torch.abs(z_fracture_2) ** 3 - torch.abs(z_fracture_2))

    rhs_value = torch.cat([rhs_fracture_1, rhs_fracture_2], dim=0)

    return rhs_value


def residual(basis, u_grad):
    """Residual functional for fracture problem."""
    u_grad = u_grad(*basis.integration_points)

    v = basis.v
    v_grad = basis.v_grad
    rhs_value = rhs(*basis.integration_points)

    return rhs_value * v - v_grad @ u_grad.mT


def gram_matrix(basis):
    """bilinear form for fracture problem."""
    v_grad = basis.v_grad

    return v_grad @ v_grad.mT


gram_matrix_inverse = torch.inverse(
    discrete_basis.reduce(discrete_basis.integrate_bilinear_form(gram_matrix))
)

Ih, Ih_grad = discrete_basis.interpolate(discrete_basis)


def interpolation_nn(_):
    """Interpolation of the Neural Network in the FEM basis."""
    return Ih(NN)


def grad_interpolation_nn(_):
    """Interpolation of the Neural Network gradient in the FEM basis."""
    return Ih_grad(NN)


# ---------------------- Error Parameters ----------------------#


def exact(x, y, z):
    """Exact solution for fracture problem."""
    x_fracture_1, _ = torch.split(x, 1, dim=0)
    y_fracture_1, y_fracture_2 = torch.split(y, 1, dim=0)
    _, z_fracture_2 = torch.split(z, 1, dim=0)

    exact_fracture_1 = (
        -y_fracture_1
        * (1 - y_fracture_1)
        * torch.abs(x_fracture_1)
        * (x_fracture_1**2 - 1)
    )
    exact_fracture_2 = (
        y_fracture_2
        * (1 - y_fracture_2)
        * torch.abs(z_fracture_2)
        * (z_fracture_2**2 - 1)
    )

    exact_value = torch.cat([exact_fracture_1, exact_fracture_2], dim=0)

    return exact_value


def exact_grad(x, y, z):
    """Gradient of the exact solution for fracture problem."""
    x_fracture_1, _ = torch.split(x, 1, dim=0)
    y_fracture_1, y_fracture_2 = torch.split(y, 1, dim=0)
    _, z_fracture_2 = torch.split(z, 1, dim=0)

    exact_dx_fracture_1 = (
        -y_fracture_1
        * (1 - y_fracture_1)
        * (
            torch.sign(x_fracture_1) * (x_fracture_1**2 - 1)
            + 2 * x_fracture_1 * torch.abs(x_fracture_1)
        )
    )
    exact_dy_fracture_1 = (
        -(1 - 2 * y_fracture_1) * torch.abs(x_fracture_1) * (x_fracture_1**2 - 1)
    )
    exact_dz_fracture_1 = torch.zeros_like(exact_dx_fracture_1)

    exact_grad_fracture_1 = torch.cat(
        [exact_dx_fracture_1, exact_dy_fracture_1, exact_dz_fracture_1], dim=-1
    )

    exact_dy_fracture_2 = (
        (1 - 2 * y_fracture_2) * torch.abs(z_fracture_2) * (z_fracture_2**2 - 1)
    )
    exact_dz_fracture_2 = (
        y_fracture_2
        * (1 - y_fracture_2)
        * (
            torch.sign(z_fracture_2) * (z_fracture_2**2 - 1)
            + 2 * z_fracture_2 * torch.abs(z_fracture_2)
        )
    )
    exact_dx_fracture_2 = torch.zeros_like(exact_dz_fracture_2)

    exact_grad_fracture_2 = torch.cat(
        [exact_dx_fracture_2, exact_dy_fracture_2, exact_dz_fracture_2], dim=-1
    )

    grad_value = torch.cat([exact_grad_fracture_1, exact_grad_fracture_2], dim=0)

    return grad_value


def h1_exact(basis):
    """H1 norm of the exact solution for fracture problem."""
    exact_value = exact(*basis.integration_points)

    exact_dx_value, exact_dy_value, exact_dz_value = torch.split(
        exact_grad(*basis.integration_points), 1, dim=-1
    )

    return exact_value**2 + exact_dx_value**2 + exact_dy_value**2 + exact_dz_value**2

    # return exact_value**2


def h1_norm(basis, solution, solution_grad):
    """H1 norm of the Neural Network solution for fracture problem."""
    exact_value = exact(*basis.integration_points)

    l2_error = (exact_value - solution(*basis.integration_points)) ** 2

    exact_dx_value, exact_dy_value, exact_dz_value = torch.split(
        exact_grad(*basis.integration_points), 1, dim=-1
    )

    solution_dx, solution_dy, solution_dz = torch.split(
        solution_grad(*basis.integration_points), 1, dim=-1
    )

    h1_0_error = (
        (exact_dx_value - solution_dx) ** 2
        + (exact_dy_value - solution_dy) ** 2
        + (exact_dz_value - solution_dz) ** 2
    )

    return h1_0_error + l2_error

    # return L2_error


exact_norm = torch.sqrt(torch.sum(discrete_basis.integrate_functional(h1_exact)))

# ---------------------- Training ----------------------#

start_time = datetime.now()


def training_step(neural_network):
    """Training step for the neural network."""
    residual_vector = discrete_basis.reduce(
        discrete_basis.integrate_linear_form(residual, neural_network.gradient)
    )

    loss_value = residual_vector.T @ (gram_matrix_inverse @ residual_vector)

    # loss_value = torch.sum(residual_vector**2, dim=0)

    relative_loss = torch.sqrt(loss_value) / exact_norm**2

    h1_error = torch.sqrt(
        torch.sum(
            discrete_basis.integrate_functional(
                h1_norm, neural_network, neural_network.gradient
            )
        )
    )

    return loss_value, relative_loss, h1_error / exact_norm


# ---------------------- Plot Values ----------------------#

### --- FEM SOLUTION PARAMETERS --- ###

local_vertices_3D = discrete_basis.global_triangulation["vertices_3D"][
    discrete_basis.global_triangulation["global2local_idx"].reshape(2, -1)
]

vertices_fracture_1, vertices_fracture_2 = torch.unbind(
    mesh.local_triangulations["vertices"], dim=0
)

triangles_fracture_1, triangles_fracture_2 = torch.unbind(
    mesh.local_triangulations["triangles"], dim=0
)

u_NN_local = NN(*torch.split(local_vertices_3D, 1, -1))

u_NN_fracture_1, u_NN_fracture_2 = torch.unbind(u_NN_local, dim=0)


### --- TRACE PARAMETERS --- ###

trace_nodes = discrete_basis.global_triangulation["vertices_3D"][
    discrete_basis.global_triangulation["traces__global_vertices_idx"], 1
].numpy(force=True)

local_vertices_3D = discrete_basis.global_triangulation["vertices_3D"][
    discrete_basis.global_triangulation["global2local_idx"].reshape(2, -1)
]

exact_value_local = exact(*torch.split(local_vertices_3D, 1, -1))

exact_value_global = exact_value_local.reshape(-1, 1)[
    discrete_basis.global_triangulation["local2global_idx"]
]

exact_trace = exact_value_global[
    discrete_basis.global_triangulation["traces__global_vertices_idx"]
].numpy(force=True)

u_NN_global = u_NN_local.reshape(-1, 1)[
    discrete_basis.global_triangulation["local2global_idx"]
]

u_NN_trace = u_NN_global[
    discrete_basis.global_triangulation["traces__global_vertices_idx"]
].numpy(force=True)

### --- JUMP PARAMETERS --- ###

V_inner_edges = InteriorEdgesFractureBasis(
    mesh, ElementLine(polynomial_order=1, integration_order=2)
)

traces_local_edges_idx = discrete_basis.global_triangulation["traces_local_edges_idx"]

n_E = discrete_basis.mesh.local_triangulations["normal4inner_edges_3D"]

n4e_u = mesh.edges_parameters["nodes4unique_edges"]

nodes4trace = n4e_u[torch.arange(n4e_u.shape[0])[:, None], traces_local_edges_idx]

c4n = discrete_basis.mesh.local_triangulations["vertices"]

coords4trace = c4n[torch.arange(c4n.shape[0]), nodes4trace]

# points_trace_fracture_1, points_trace_fracture_2 = torch.unbind(, dim = 0)

sort_points_trace, sort_idx = torch.sort(coords4trace.mean(-2)[..., 1])

points_trace_fracture_1, points_trace_fracture_2 = torch.unbind(
    sort_points_trace, dim=0
)

# COMPUTE JUMP neural_network SOLUTION

_, I_E_grad = discrete_basis.interpolate(V_inner_edges)

I_E_NN_grad = I_E_grad(NN)

I_E_u_NN_grad_K_plus, I_E_u_NN_grad_minus = torch.unbind(I_E_NN_grad, dim=-4)

jump_u_NN = (I_E_u_NN_grad_K_plus * n_E).sum(-1) + (I_E_u_NN_grad_minus * -n_E).sum(-1)

jump_u_NN_trace = jump_u_NN[
    torch.arange(jump_u_NN.shape[0])[:, None], traces_local_edges_idx
]

sort_jump_u_NN_trace = jump_u_NN_trace[
    torch.arange(jump_u_NN_trace.shape[0])[:, None], sort_idx
]

jump_u_NN_trace_fracture_1, jump_u_NN_trace_fracture_2 = torch.unbind(
    sort_jump_u_NN_trace, dim=0
)

# COMPUTE JUMP EXACT

local_vertices_3D = discrete_basis.global_triangulation["vertices_3D"][
    discrete_basis.global_triangulation["global2local_idx"].reshape(2, -1)
]

exact_value_local = exact(*torch.split(local_vertices_3D, 1, -1))

exact_value_global = exact_value_local.reshape(-1, 1)[
    discrete_basis.global_triangulation["local2global_idx"]
]

I_E_u, I_E_u_grad = discrete_basis.interpolate(V_inner_edges, exact_value_global)

I_E_u_grad_K_plus, I_E_u_grad_minus = torch.unbind(I_E_u_grad, dim=-4)

n_E = discrete_basis.mesh.local_triangulations["normal4inner_edges_3D"]

jump_u = (I_E_u_grad_K_plus * n_E).sum(-1) + (I_E_u_grad_minus * -n_E).sum(-1)

jump_u_trace = jump_u[torch.arange(jump_u.shape[0])[:, None], traces_local_edges_idx]

sort_jump_u_trace = jump_u_trace[torch.arange(jump_u_trace.shape[0])[:, None], sort_idx]

jump_u_trace_fracture_1, jump_u_trace_fracture_2 = torch.unbind(
    sort_jump_u_trace, dim=0
)

### --- ERROR PARAMETERS --- ###

numerator = discrete_basis.integrate_functional(
    h1_norm, interpolation_nn, grad_interpolation_nn
)
denominator = discrete_basis.integrate_functional(h1_exact)

EPSILON = 1e-10
safe_denominator = torch.where(
    denominator.abs() < EPSILON, torch.ones_like(denominator), denominator
)

H1_error_fracture_1, H1_error_fracture_2 = torch.unbind(
    torch.sqrt(numerator / safe_denominator), dim=0
)

# H1_error_fracture_1, H1_error_fracture_2 = torch.unbind(
#     torch.sqrt(
#         V.integrate_functional(H1_norm, interpolation_nn, grad_interpolation_nn)
#         / V.integrate_functional(H1_exact)
#     ),
#     dim=0,
# )

print(
    torch.sqrt(
        torch.sum(
            discrete_basis.integrate_functional(
                h1_norm, interpolation_nn, grad_interpolation_nn
            )
        )
    )
    / exact_norm
)

c4e_fracture_1, c4e_fracture_2 = torch.unbind(
    mesh.local_triangulations["coords4triangles"], dim=0
)

# ---------------------- Plot ---------------------- #


# NAME = "non_interpolated_vpinns"
# NAME = "vpinns"
NAME = "rvpinns"
# NAME = "vpinns_tanh"
SAVE_DIR = "figures"
os.makedirs(SAVE_DIR, exist_ok=True)

# ------------------ neural_network SOLUTION ------------------

# # Fracture 1
# fig = plt.figure(dpi=200)
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_trisurf(vertices_fracture_1.numpy(force=True)[:, 0],
#                 vertices_fracture_1.numpy(force=True)[:, 1],
#                 u_NN_fracture_1.reshape(-1).numpy(force=True),
#                 triangles=triangles_fracture_1.numpy(force=True),
#                 cmap='viridis', edgecolor='black', linewidth=0.1)
# # ax.set_title("Fracture 1")
# ax.set_xlabel(r"$x$")
# ax.set_ylabel(r"$y$")
# ax.set_zlabel(r"$u_h(x,y)$")
# ax.tick_params(labelsize=8)
# plt.tight_layout()
# plt.savefig(os.path.join(SAVE_DIR, f"{NAME}_nn_solution_fracture_1.png"))
#

# # Fracture 2
# fig = plt.figure(dpi=200)
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_trisurf(vertices_fracture_2.numpy(force=True)[:, 0],
#                 vertices_fracture_2.numpy(force=True)[:, 1],
#                 u_NN_fracture_2.reshape(-1).numpy(force=True),
#                 triangles=triangles_fracture_2.numpy(force=True),
#                 cmap='viridis', edgecolor='black', linewidth=0.1)
# # ax.set_title("Fracture 2")
# ax.set_xlabel(r"$x$")
# ax.set_ylabel(r"$y$")
# ax.set_zlabel(r"$u_h(x,y)$")
# ax.tick_params(labelsize=8)
# plt.tight_layout()
# plt.savefig(os.path.join(SAVE_DIR, f"{NAME}_nn_solution_fracture_2.png"))
#

vertices = torch.unbind(mesh.local_triangulations["vertices_3D"], dim=0)

triangles = torch.unbind(mesh.local_triangulations["triangles"], dim=0)

u_h = torch.unbind(u_NN_local, dim=0)

plotter = pv.Plotter(off_screen=True)

for i in range(2):
    vertices_numpy = vertices[i].numpy(force=True)  # (N_v, 3)
    triangles_numpy = triangles[i].numpy(force=True)  # (N_T, 3)
    solution_numpy = u_h[i].numpy(force=True)

    # Pyvista expects a flat list with a prefix indicating
    # the number of vertices per cell (3 for triangles)
    faces = np.hstack(
        [np.full((triangles_numpy.shape[0], 1), 3), triangles_numpy]
    ).flatten()

    mesh = pv.PolyData(vertices_numpy, faces)

    mesh.point_data["solution"] = solution_numpy

    plotter.add_mesh(
        mesh,
        show_edges=True,
        scalars="solution",
        cmap="viridis",
        opacity=1,
        scalar_bar_args={"title": "Pressure"},
        lighting=False,
    )

plotter.show()
plotter.screenshot(os.path.join(SAVE_DIR, f"{NAME}_solution.png"))

# ------------------ TRACES (JUMPS) ------------------

plt.figure(dpi=200)

plt.plot(trace_nodes, u_NN_trace, linestyle="--")

plt.xlabel("trace length")
plt.ylabel("value")
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, f"{NAME}_trace_value_nn.png"))

# ------------------ TRACES (JUMPS) ------------------

# Fracture 1
fig = plt.figure(dpi=200)
plt.plot(
    points_trace_fracture_1.numpy(force=True),
    jump_u_trace_fracture_1.reshape(-1).numpy(force=True),
    color="black",
    label=r"$u^{ex}$",
)
plt.scatter(
    points_trace_fracture_1.numpy(force=True),
    jump_u_NN_trace_fracture_1.reshape(-1).numpy(force=True),
    color="r",
    label=r"$u_h$",
)
plt.xlabel("trace length")
plt.ylabel("jump value")
# plt.title("Fracture 1")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, f"{NAME}_trace_jump_fracture_1.png"))

# Fracture 2
fig = plt.figure(dpi=200)
plt.plot(
    points_trace_fracture_2.numpy(force=True),
    jump_u_trace_fracture_2.reshape(-1).numpy(force=True),
    color="black",
    label=r"$u^{ex}$",
)
plt.scatter(
    points_trace_fracture_2.numpy(force=True),
    jump_u_NN_trace_fracture_2.reshape(-1).numpy(force=True),
    color="r",
    label=r"$u_h$",
)
plt.xlabel("trace length")
plt.ylabel("jump value")
# plt.title("Fracture 2")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, f"{NAME}_trace_jump_fracture_2.png"))
