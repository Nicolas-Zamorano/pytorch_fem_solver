"""Example of VPINNs for the 3D fracture problem with convergence study"""

import os
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensordict as td
import torch
import triangle as tr
from fem import ElementTri, FractureBasis, Fractures

from neural_network import NeuralNetwork3D

torch.set_default_dtype(torch.float64)
# torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
# torch.cuda.empty_cache()

# ---------------------- Neural Network Functions ----------------------#


def nn_gradient(neural_net, x, y, z):
    """Compute gradient of a Neural Network w.r.t inputs."""
    x.requires_grad_(True)
    y.requires_grad_(True)
    z.requires_grad_(True)

    output = neural_net.forward(x, y, z)

    gradients = torch.autograd.grad(
        outputs=output,
        inputs=(x, y, z),
        grad_outputs=torch.ones_like(output),
        retain_graph=True,
        create_graph=True,
    )

    return torch.concat(gradients, dim=-1)


def optimizer_step(opt, loss):
    """Perform one step of the optimizer."""
    opt.zero_grad()
    loss.backward(retain_graph=True)
    opt.step()
    scheduler.step()


# ---------------------- Neural Network Parameters ----------------------#

EPOCHS = 5000
LEARNING_RATE = 0.2e-3
DECAY_RATE = 0.98
DECAY_STEPS = 200

NN = torch.jit.script(
    NeuralNetwork3D(
        input_dimension=3,
        output_dimension=1,
        deep_layers=4,
        hidden_layers_dimension=15,
        activation_function=torch.nn.ReLU(),
    )
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


for i in range(11):

    # torch.cuda.empty_cache()
    NN.load_state_dict(NN_initial_parameters)

    optimizer = torch.optim.Adam(NN.parameters(), lr=LEARNING_RATE)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, DECAY_RATE ** (1 / DECAY_STEPS)
    )

    fracture_triangulation = tr.triangulate(
        fracture_triangulation, "prsea" + str(MESH_SIZE ** (EXPONENT + i))
    )

    fracture_triangulation_torch = td.TensorDict((fracture_triangulation))

    fractures_triangulation = (
        fracture_triangulation_torch,
        fracture_triangulation_torch,
    )

    mesh = Fractures(
        triangulations=fractures_triangulation, fractures_3d_data=fractures_data
    )

    elements = ElementTri(polynomial_order=1, integration_order=2)

    V = FractureBasis(mesh, elements)

    exact_H1_norm = torch.sqrt(torch.sum(V.integrate_functional(h1_exact)))

    A = V.reduce(V.integrate_bilinear_form(a))

    A_inv = torch.linalg.inv(A)

    Ih, Ih_grad = V.interpolate(V)

    exact_norm = torch.sqrt(torch.sum(V.integrate_functional(h1_exact)))

    LOSS_OPT = 10e4
    params_opt = NN.state_dict()

    for epoch in range(EPOCHS):
        current_time = datetime.now().strftime("%H:%M:%S")
        print(f"{'='*15} [{current_time}] Iter:{i} Epoch:{epoch + 1}/{EPOCHS} {'='*15}")
        residual_value = V.reduce(
            V.integrate_linear_form(residual, grad_interpolation_nn)
        )

        # loss_value = residual_value.T @ (A_inv @ residual_value)

        loss_value = (residual_value**2).sum()

        print(f"Loss: {loss_value.item():.8f}")

        optimizer_step(optimizer, loss_value)

        if loss_value < LOSS_OPT:
            LOSS_OPT = loss_value
            params_opt = NN.state_dict()

    NN.load_state_dict(params_opt)

    H1_norm_value = (
        torch.sqrt(
            torch.sum(
                V.integrate_functional(h1_norm, interpolation_nn, grad_interpolation_nn)
            )
        )
        / exact_norm
    )

    H1_norm_list.append((H1_norm_value).item())

    nb_dofs_list.append(V.basis_parameters["nb_dofs"])

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
