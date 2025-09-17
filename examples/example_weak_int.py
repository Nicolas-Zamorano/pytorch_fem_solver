"""Example of weak formulation using FEM and Neural Networks with interpolation."""

import math
from datetime import datetime

import matplotlib.pyplot as plt
import skfem
import torch

from fem import Basis, ElementTri, MeshTri
from neural_network import NeuralNetwork

# torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
# torch.cuda.empty_cache()
# torch.set_default_dtype(torch.float64)

# ---------------------- Neural Network Functions ----------------------#


def nn_gradient(neural_net, x, y):
    """Compute gradient of a Neural Network w.r.t inputs."""
    x.requires_grad_(True)
    y.requires_grad_(True)

    output = neural_net.forward(x, y)

    gradients = torch.autograd.grad(
        outputs=output,
        inputs=(x, y),
        grad_outputs=torch.ones_like(output),
        retain_graph=True,
        create_graph=True,
    )

    return torch.concat(gradients, dim=-1)


def optimizer_step(opt, scheduler_func, value_loss):
    """Perform an optimization step."""
    opt.zero_grad()
    value_loss.backward(retain_graph=True)
    opt.step()
    scheduler_func.step()


# ---------------------- Neural Network Parameters ----------------------#

EPOCHS = 5000
LEARNING_RATE = 0.5e-3
DECAY_RATE = 0.99
DECAY_STEPS = 100

NN = torch.jit.script(
    NeuralNetwork(
        input_dimension=2, output_dimension=1, deep_layers=4, hidden_layers_dimension=20
    )
)

NN_int = torch.jit.script(
    NeuralNetwork(
        input_dimension=2, output_dimension=1, deep_layers=4, hidden_layers_dimension=20
    )
)

optimizer = torch.optim.Adam(NN.parameters(), lr=LEARNING_RATE)

optimizer_int = torch.optim.Adam(NN_int.parameters(), lr=LEARNING_RATE)

scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optimizer, DECAY_RATE ** (1 / DECAY_STEPS)
)

scheduler_int = torch.optim.lr_scheduler.ExponentialLR(
    optimizer_int, DECAY_RATE ** (1 / DECAY_STEPS)
)

# ---------------------- FEM Parameters ----------------------#

K_REF = 3

Q = 1
K_INT = 2
K_TEST = 1

mesh_sk_H = skfem.MeshTri1().refined(K_REF)

mesh_sk_h = mesh_sk_H.refined(K_TEST)

V_H = Basis(
    MeshTri(torch.tensor(mesh_sk_H.p).T, torch.tensor(mesh_sk_H.t).T),
    ElementTri(polynomial_order=K_INT, integration_order=Q),
)

V_h = Basis(
    MeshTri(torch.tensor(mesh_sk_h.p).T, torch.tensor(mesh_sk_h.t).T),
    ElementTri(polynomial_order=K_TEST, integration_order=Q),
)

interpolation_func, grad_interpolation_func = V_H.interpolate(V_h)

# ---------------------- Residual Parameters ----------------------#


def nn_grad_func(x, y):
    """Compute gradient of the primary Neural Network."""
    return nn_gradient(NN, x, y)


def interpolate_nn(_):
    """Interpolate the primary Neural Network using FEM basis."""
    return interpolation_func(NN)


def interpolate_nn_grad(_):
    """Interpolate the gradient of the primary Neural Network using FEM basis."""
    return grad_interpolation_func(NN)


def rhs(x, y):
    """Right-hand side function."""
    return 2.0 * math.pi**2 * torch.sin(math.pi * x) * torch.sin(math.pi * y)


def residual(basis, nn_grad):
    """Compute the residual of the Poisson equation using the neural network as an approximation."""
    x, y = basis.integration_points

    nn_grad = nn_grad(x, y)

    v = basis.v
    v_grad = basis.v_grad
    rhs_value = rhs(x, y)

    return rhs_value * v - (v_grad @ nn_grad.mT)


# def gram_matrix(elements: Elements):

#     v = elements.v
#     v_grad = elements.v_grad

#     return v_grad @ v_grad.mT + v @ v.mT

# A = V_h.integrate_bilinear_form(gram_matrix)[V_h.inner_dofs, :][:, V_h.inner_dofs]

# A_inv = torch.linalg.inv(A)

# ---------------------- Error Parameters ----------------------#


def exact(x, y):
    """Exact solution."""
    return torch.sin(math.pi * x) * torch.sin(math.pi * y)


def exact_dx(x, y):
    """Exact solution derivative w.r.t x."""
    return math.pi * torch.cos(math.pi * x) * torch.sin(math.pi * y)


def exact_dy(x, y):
    """Exact solution derivative w.r.t y."""
    return math.pi * torch.sin(math.pi * x) * torch.cos(math.pi * y)


def h1_exact(basis):
    """Compute the H1 norm of the exact solution."""
    x, y = basis.integration_points

    return exact_dx(x, y) ** 2 + exact_dy(x, y) ** 2 + exact(x, y) ** 2


def h1_norm(basis):
    """Compute the H1 norm of the neural network solution."""
    x, y = basis.integration_points

    nn_dx, nn_dy = torch.split(nn_gradient(NN, x, y), 1, dim=-1)

    l2_error = (exact(x, y) - NN(x, y)) ** 2

    h1_0_error = (exact_dx(x, y) - nn_dx) ** 2 + (exact_dy(x, y) - nn_dy) ** 2

    return l2_error + h1_0_error


exact_norm = torch.sqrt(torch.sum(V_h.integrate_functional(h1_exact)))

H_1_error_int_list = []
H1_error_list = []

# ---------------------- Training ----------------------#

start_time = datetime.now()

for epoch in range(EPOCHS):
    current_time = datetime.now().strftime("%H:%M:%S")
    print(f"{'='*20} [{current_time}] Epoch:{epoch + 1}/{EPOCHS} {'='*20}")

    residual_value = V_h.reduce(V_h.integrate_linear_form(residual, nn_grad_func))

    # loss_value = residual_value.T @ (A_inv @ residual_value)

    loss_value = (residual_value**2).sum()

    optimizer_step(optimizer, scheduler, loss_value)

    error_norm = (
        torch.sqrt(torch.sum(V_h.integrate_functional(h1_norm, NN, nn_grad_func)))
        / exact_norm
    )

    residual_value_int = V_h.reduce(
        V_h.integrate_linear_form(residual, interpolate_nn_grad)
    )
    # loss_value_int = residual_value_int.T @ (A_inv @ residual_value_int)

    loss_value_int = (residual_value_int**2).sum()

    optimizer_step(optimizer_int, scheduler_int, loss_value_int)

    error_norm_int = (
        torch.sqrt(
            torch.sum(
                V_h.integrate_functional(h1_norm, interpolate_nn, interpolate_nn_grad)
            )
        )
        / exact_norm
    )

    print(
        f"u_NN error: {error_norm.item():.8f} I_H u_NN error: {error_norm_int.item():.8f}"
    )

    H1_error_list.append(error_norm.item())
    H_1_error_int_list.append(error_norm_int.item())


end_time = datetime.now()

execution_time = end_time - start_time

print(f"Training time: {execution_time}")

# ---------------------- Plotting ----------------------#

NB_PLOT_POINTS = 100

X, Y = torch.meshgrid(
    torch.linspace(0, 1, NB_PLOT_POINTS),
    torch.linspace(0, 1, NB_PLOT_POINTS),
    indexing="ij",
)

with torch.no_grad():
    Z = abs(torch.sin(math.pi * X) * torch.sin(math.pi * Y) - NN_int(X, Y))

figure_solution, axis_solution = plt.subplots()

fig, ax = plt.subplots(dpi=500)
c = ax.contourf(X.cpu(), Y.cpu(), Z.cpu(), levels=100, cmap="viridis")
fig.colorbar(c, ax=ax, orientation="vertical")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title(r"$|u-I_H u_\theta|$")
plt.tight_layout()

figure_error, axis_error = plt.subplots(dpi=500)

figure_error, axis_error = plt.subplots(dpi=500)

axis_error.semilogy(
    H1_error_list, label=r"$\frac{\|u-u_\theta\|_{U}}{\|u\|_{U}}$", linestyle="-."
)

axis_error.semilogy(
    H_1_error_int_list,
    label=r"$\frac{\|u-I_H u_\theta\|_{U}}{\|u\|_{U}}$",
    linestyle=":",
)

axis_error.legend(fontsize=15)

plt.show()
