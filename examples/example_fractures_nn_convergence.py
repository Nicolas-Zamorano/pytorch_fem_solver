"""Example of VPINNs for the 3D fracture problem with convergence study"""

import os
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensordict as td
import torch
import triangle as tr
from torch_fem import (
    ElementTri,
    FractureBasis,
    FracturesTri,
    FeedForwardNeuralNetwork,
    Model,
)

torch.set_default_dtype(torch.float64)
# torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
# torch.cuda.empty_cache()

# ---------------------- Neural Network Parameters ----------------------#


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

MESH_SIZE = 0.5

EXPONENT = 3

fracture_2d_data = {
    "vertices": [
        [-1.0, 0.0],
        [1.0, 0.0],
        [-1.0, 1.0],
        [1.0, 1.0],
        [0.0, 0.0],
        [0.0, 1.0],
    ],
    "segments": [[0, 1], [1, 3], [2, 3], [0, 2], [4, 5]],
}

fractures_data = torch.tensor(
    [
        [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        [[0.0, 0.0, -1.0], [0.0, 0.0, 1.0], [0.0, 1.0, -1.0], [0.0, 1.0, 1.0]],
    ]
)

fracture_triangulation = tr.triangulate(
    fracture_2d_data, "pqsea" + str(MESH_SIZE ** (EXPONENT))
)

# ---------------------- Residual Parameters ----------------------#


def rhs(x, y, z):
    """Right-hand side function."""

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


def residual(basis, solution_grad):
    """Residual of the PDE."""

    v = basis.v
    v_grad = basis.v_grad
    rhs_value = rhs(*basis.integration_points)

    return rhs_value * v - v_grad @ solution_grad(*basis.integration_points).mT


def a(basis):
    """Bilinear form."""
    v_grad = basis.v_grad

    return v_grad @ v_grad.mT


# ---------------------- Error Parameters ----------------------#


def exact(x, y, z):
    """Exact solution."""
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
    """Gradient of the exact solution."""
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
    """Exact H1 norm."""
    exact_value = exact(*basis.integration_points)

    exact_dx_value, exact_dy_value, exact_dz_value = torch.split(
        exact_grad(*basis.integration_points), 1, dim=-1
    )

    return exact_value**2 + exact_dx_value**2 + exact_dy_value**2 + exact_dz_value**2


def h1_norm(basis, solution, solution_grad):
    """H1 norm of the error."""
    exact_value = exact(*basis.integration_points)

    solution_value = solution(*basis.integration_points)

    l2_error = (exact_value - solution_value) ** 2

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


NN_initial_parameters = NN.state_dict()

# ---------------------- Solution ----------------------#

H1_norm_list = []
nb_dofs_list = []


def interpolation_nn(_):
    """Interpolation of the Neural Network in the FEM basis."""
    return Ih(NN)


def grad_interpolation_nn(_):
    """Interpolation of the Neural Network gradient in the FEM basis."""
    return Ih_grad(NN)


def training_step(neural_network: FeedForwardNeuralNetwork):
    """Training step for the neural network."""
    residual_vector = V.reduce(
        # discrete_basis.integrate_linear_form(residual, neural_network.gradient)
        V.integrate_linear_form(residual, grad_interpolation_nn)
    )

    loss_value = residual_vector.T @ (A @ residual_vector)

    # loss_value = torch.sum(residual_vector**2, dim=0)

    relative_loss = torch.sqrt(loss_value) / exact_norm**2

    h1_error = torch.sqrt(
        torch.sum(
            V.integrate_functional(
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
    # learning_rate_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
    # scheduler_kwargs={"gamma": 0.99**100},
    use_early_stopping=True,
    early_stopping_patience=50,
    min_delta=1e-12,
)

model.train()

for i in range(11):

    # torch.cuda.empty_cache()
    NN.load_state_dict(NN_initial_parameters)

    fracture_triangulation = tr.triangulate(
        fracture_triangulation, "prsea" + str(MESH_SIZE ** (EXPONENT + i))
    )

    fracture_triangulation_torch = td.TensorDict((fracture_triangulation))

    fractures_triangulation = [
        fracture_triangulation,
        fracture_triangulation,
    ]

    mesh = FracturesTri(
        triangulations=fractures_triangulation, fractures_3d_data=fractures_data
    )

    elements = ElementTri(polynomial_order=1, integration_order=2)

    V = FractureBasis(mesh, elements)

    exact_H1_norm = torch.sqrt(torch.sum(V.integrate_functional(h1_exact)))

    A = V.reduce(V.integrate_bilinear_form(a))

    A_inv = torch.linalg.inv(A)  # pylint: disable=not-callable

    Ih, Ih_grad = V.interpolate(V)

    exact_norm = torch.sqrt(torch.sum(V.integrate_functional(h1_exact)))

    model.train()

    model.load_optimal_parameters()

    H1_norm_value = (
        torch.sqrt(
            torch.sum(
                V.integrate_functional(h1_norm, interpolation_nn, grad_interpolation_nn)
            )
        )
        / exact_norm
    )

    H1_norm_list.append((H1_norm_value).item())

    nb_dofs_list.append(A.shape[-1])

# ------------------ CONFIG ------------------

NAME = "vpinns"
# name = "rvpinns"
# name = "non_interpolated_vpinns"
SAVE_DIR = "figures"
os.makedirs(SAVE_DIR, exist_ok=True)

# ------------------ CONVERGENCE VPINNs ------------------

nb_dofs_np = np.array(nb_dofs_list)
H1_norm_np = np.array(H1_norm_list)

log_dofs = np.log10(nb_dofs_np)
log_H1 = np.log10(H1_norm_np)
slope, intercept = np.polyfit(log_dofs, log_H1, 1)

H1_fit = 10**intercept * nb_dofs_np**slope

fig, ax = plt.subplots(dpi=200)
ax.loglog(
    nb_dofs_list,
    H1_norm_np,
    "^",
    color="orange",
    markersize=7,
    markeredgecolor="black",
    label=f"decay rate = {-slope:.2f}",
)

ax.loglog(nb_dofs_list, H1_fit, "-.", color="orange", alpha=0.5)

ax.set_xlabel("# DOFs")
ax.set_ylabel(r"$H^1$ Relative Error")
ax.legend()
ax.grid(True, which="both", linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, f"{NAME}_H1_convergence.png"))
plt.show()

# ------------------ SAVE DATA ------------------

with open(os.path.join(SAVE_DIR, f"{NAME}_H1_norm_convergence.pkl"), "wb") as file:
    pickle.dump([nb_dofs_np, H1_norm_np], file)
