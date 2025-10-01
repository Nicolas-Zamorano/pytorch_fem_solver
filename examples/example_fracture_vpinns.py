"""Example of Variational Physics-Informed Neural Networks (VPINNs) for a 3D fracture problem."""

import os

import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import torch
import triangle as tr

from matplotlib.collections import PolyCollection

from torch_fem import (
    ElementLine,
    ElementTri,
    FractureBasis,
    FracturesTri,
    InteriorEdgesFractureBasis,
    Model,
    FeedForwardNeuralNetwork,
)

torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
torch.set_default_dtype(torch.float64)


class BoundaryConstrain(torch.nn.Module):
    """Class to strongly apply bc"""

    def forward(self, inputs):
        """Boundary condition modifier function."""
        x, y, z = torch.split(inputs, 1, dim=-1)
        return (x + 1) * (x - 1) * y * (y - 1) * (z + 1) * (z - 1)


NN = FeedForwardNeuralNetwork(
    input_dimension=3,
    output_dimension=1,
    nb_hidden_layers=4,
    neurons_per_layers=15,
    activation_function=torch.nn.ReLU(),
    boundary_condition_modifier=BoundaryConstrain(),
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

fracture_triangulation = tr.triangulate(fracture_2d_data, "pqsena" + str(0.5**8))

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


def rhs(coordinates):
    """Right-hand side function for facture problem."""
    x, y, z = torch.split(coordinates, 1, dim=-1)
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
    integration_points = basis.integration_points
    u_grad = u_grad(integration_points)

    v = basis.v
    v_grad = basis.v_grad
    rhs_value = rhs(integration_points)

    return rhs_value * v - v_grad @ u_grad.mT


def gram_matrix(basis):
    """bilinear form for fracture problem."""
    v_grad = basis.v_grad

    return v_grad @ v_grad.mT


gram_matrix_inverse = torch.inverse(
    discrete_basis.reduce(discrete_basis.integrate_bilinear_form(gram_matrix))
)

Ih, Ih_grad = discrete_basis.interpolate(discrete_basis)


def interpolation_nn():
    """Interpolation of the Neural Network in the FEM basis."""
    return Ih(NN)


def grad_interpolation_nn(_):
    """Interpolation of the Neural Network gradient in the FEM basis."""
    return Ih_grad(NN)


# ---------------------- Error Parameters ----------------------#


def exact(coordinates):
    """Exact solution for fracture problem."""
    x, y, z = torch.split(coordinates, 1, dim=-1)
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


def exact_grad(coordinates):
    """Gradient of the exact solution for fracture problem."""
    x, y, z = torch.split(coordinates, 1, dim=-1)
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
    integration_points = basis.integration_points
    exact_value = exact(integration_points)

    exact_dx_value, exact_dy_value, exact_dz_value = torch.split(
        exact_grad(integration_points), 1, dim=-1
    )

    return exact_value**2 + exact_dx_value**2 + exact_dy_value**2 + exact_dz_value**2

    # return exact_value**2


def h1_norm(basis, solution, solution_grad):
    """H1 norm of the Neural Network solution for fracture problem."""
    integration_points = basis.integration_points

    exact_value = exact(integration_points)

    l2_error = (exact_value - solution(integration_points)) ** 2

    exact_dx_value, exact_dy_value, exact_dz_value = torch.split(
        exact_grad(integration_points), 1, dim=-1
    )

    solution_dx, solution_dy, solution_dz = torch.split(
        solution_grad(integration_points), 1, dim=-1
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


def training_step(neural_network):
    """Training step for the neural network."""
    residual_vector = discrete_basis.reduce(
        # discrete_basis.integrate_linear_form(residual, neural_network.gradient)
        discrete_basis.integrate_linear_form(residual, grad_interpolation_nn)
    )

    loss_value = residual_vector.T @ (gram_matrix_inverse @ residual_vector)

    # loss_value = torch.sum(residual_vector**2, dim=0)

    relative_loss = torch.sqrt(loss_value) / exact_norm**2

    h1_error = torch.sqrt(
        torch.sum(
            discrete_basis.integrate_functional(
                # h1_norm, neural_network, neural_network.gradient
                h1_norm,
                interpolation_nn,
                grad_interpolation_nn,
            )
        )
    )

    return loss_value, relative_loss, h1_error / exact_norm


model = Model(
    neural_network=NN,
    training_step=training_step,
    epochs=1,
    optimizer=torch.optim.Adam,
    optimizer_kwargs={"lr": 0.2e-3},
    learning_rate_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
    # scheduler_kwargs={"gamma": 0.99**100},
    use_early_stopping=True,
    early_stopping_patience=50,
    min_delta=1e-12,
)

model.train()

model.plot_training_history()

# ---------------------- Plot Values ----------------------#

### --- FEM SOLUTION PARAMETERS --- ###

local_vertices_3D = discrete_basis.global_triangulation["vertices_3D"][
    discrete_basis.global_triangulation["global2local_idx"].reshape(2, -1)
]

vertices_fracture_1, vertices_fracture_2 = torch.unbind(
    mesh["vertices", "coordinates_3d"], dim=0
)

triangles_fracture_1, triangles_fracture_2 = torch.unbind(
    mesh["cells", "vertices"], dim=0
)

u_NN_local = NN(local_vertices_3D)

u_NN_fracture_1, u_NN_fracture_2 = torch.unbind(u_NN_local, dim=0)

### --- TRACE PARAMETERS --- ###

trace_nodes = torch.Tensor.numpy(
    discrete_basis.global_triangulation["vertices_3D"][
        discrete_basis.global_triangulation["traces__global_vertices_idx"], 1
    ],
    force=True,
)

local_vertices_3D = discrete_basis.global_triangulation["vertices_3D"][
    discrete_basis.global_triangulation["global2local_idx"].reshape(2, -1)
]

exact_value_local = exact(local_vertices_3D)

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

n_E = discrete_basis.mesh["interior_edges", "normals_3d"]

n4e_u = discrete_basis.mesh["edges", "vertices"]

nodes4trace = n4e_u[torch.arange(n4e_u.shape[0])[:, None], traces_local_edges_idx]

c4n = discrete_basis.mesh["vertices", "coordinates"]

coords4trace = c4n[torch.arange(c4n.shape[0]), nodes4trace]

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

n_E = discrete_basis.mesh["interior_edges", "normals_3d"]

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

c4e_fracture_1, c4e_fracture_2 = torch.unbind(mesh["cells", "coordinates"], dim=0)

# ---------------------- Plot ---------------------- #


# NAME = "non_interpolated_vpinns"
# NAME = "vpinns"
NAME = "rvpinns"
# NAME = "vpinns_tanh"
SAVE_DIR = "figures"
os.makedirs(SAVE_DIR, exist_ok=True)

# ------------------ neural_network SOLUTION ------------------

# Fracture 1
fig = plt.figure(dpi=200)
ax = fig.add_subplot(111, projection="3d")
ax.plot_trisurf(
    vertices_fracture_1.numpy(force=True)[:, 0],
    vertices_fracture_1.numpy(force=True)[:, 1],
    u_NN_fracture_1.reshape(-1).numpy(force=True),
    triangles=triangles_fracture_1.numpy(force=True),
    cmap="viridis",
    edgecolor="black",
    linewidth=0.1,
)
# ax.set_title("Fracture 1")
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.set_zlabel(r"$u_h(x,y)$")
ax.tick_params(labelsize=8)
plt.tight_layout()
# plt.savefig(os.path.join(SAVE_DIR, f"{NAME}_nn_solution_fracture_1.png"))
#

# Fracture 2
fig = plt.figure(dpi=200)
ax = fig.add_subplot(111, projection="3d")
ax.plot_trisurf(
    vertices_fracture_2.numpy(force=True)[:, 0],
    vertices_fracture_2.numpy(force=True)[:, 1],
    u_NN_fracture_2.reshape(-1).numpy(force=True),
    triangles=triangles_fracture_2.numpy(force=True),
    cmap="viridis",
    edgecolor="black",
    linewidth=0.1,
)
# ax.set_title("Fracture 2")
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.set_zlabel(r"$u_h(x,y)$")
ax.tick_params(labelsize=8)
plt.tight_layout()
# plt.savefig(os.path.join(SAVE_DIR, f"{NAME}_nn_solution_fracture_2.png"))
#

vertices = torch.unbind(mesh["vertices", "coordinates_3d"], dim=0)

triangles = torch.unbind(mesh["cells", "vertices"], dim=0)

u_h = torch.unbind(u_NN_local, dim=0)

plotter = pv.Plotter()

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
# plt.savefig(os.path.join(SAVE_DIR, f"{NAME}_trace_value_nn.png"))

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
plt.title("Fracture 1")
plt.legend()
plt.tight_layout()
# plt.savefig(os.path.join(SAVE_DIR, f"{NAME}_trace_jump_fracture_1.png"))

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
plt.title("Fracture 2")
plt.legend()
plt.tight_layout()
# plt.savefig(os.path.join(SAVE_DIR, f"{NAME}_trace_jump_fracture_2.png"))

### --- PLOT ERROR --- ###

H1_error_fracture_1 = H1_error_fracture_1.numpy(force=True)
c4e_fracture_1 = c4e_fracture_1.numpy(force=True)

H1_error_fracture_2 = H1_error_fracture_2.numpy(force=True)
c4e_fracture_2 = c4e_fracture_2.numpy(force=True)

all_errors = np.concatenate([H1_error_fracture_1, H1_error_fracture_2])
norm = colors.Normalize(vmin=all_errors.min(), vmax=all_errors.max())
cmap = cm.get_cmap("viridis")

fig_error, axes = plt.subplots(1, 2, figsize=(12, 3), dpi=200)

fig_error.suptitle("Relative error for FEM solution", fontsize=14)


# Fracture 1
face_colors_1 = cmap(norm(H1_error_fracture_1))
collection1 = PolyCollection(
    c4e_fracture_1, facecolors=face_colors_1, edgecolors="black", linewidths=0.2
)
ax1 = axes[0]
ax1.add_collection(collection1)
ax1.autoscale()
ax1.set_aspect("equal")
ax1.set_title("Fracture 1")
ax1.set_xlim([-1, 1])
ax1.set_ylim([0, 1])

# Fracture 2
face_colors_2 = cmap(norm(H1_error_fracture_2))
collection2 = PolyCollection(
    c4e_fracture_2, facecolors=face_colors_2, edgecolors="black", linewidths=0.2
)
ax2 = axes[1]
ax2.add_collection(collection2)
ax2.autoscale()
ax2.set_aspect("equal")
ax2.set_title("Fracture 2")
ax2.set_xlim([-1, 1])
ax2.set_ylim([0, 1])

sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array(all_errors)
color_bar = fig_error.colorbar(
    sm, ax=axes.ravel().tolist(), orientation="vertical", label="error"
)

plt.show()
