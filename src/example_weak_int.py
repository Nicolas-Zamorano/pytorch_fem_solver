import torch
import math

import matplotlib.pyplot as plt

from Neural_Network import Neural_Network
from fem import MeshTri, ElementTri, Basis
from datetime import datetime

import skfem

# torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
# torch.cuda.empty_cache()
# torch.set_default_dtype(torch.float64)

# ---------------------- Neural Network Functions ----------------------#


def NN_gradiant(NN, x, y):

    x.requires_grad_(True)
    y.requires_grad_(True)

    output = NN.forward(x, y)

    gradients = torch.autograd.grad(
        outputs=output,
        inputs=(x, y),
        grad_outputs=torch.ones_like(output),
        retain_graph=True,
        create_graph=True,
    )

    return torch.concat(gradients, dim=-1)


def optimizer_step(optimizer, scheduler, loss_value):
    optimizer.zero_grad()
    loss_value.backward(retain_graph=True)
    optimizer.step()
    scheduler.step()


# ---------------------- Neural Network Parameters ----------------------#

epochs = 5000
learning_rate = 0.5e-3
decay_rate = 0.99
decay_steps = 100

NN = torch.jit.script(
    Neural_Network(
        input_dimension=2, output_dimension=1, deep_layers=4, hidden_layers_dimension=20
    )
)

NN_int = torch.jit.script(
    Neural_Network(
        input_dimension=2, output_dimension=1, deep_layers=4, hidden_layers_dimension=20
    )
)

optimizer = torch.optim.Adam(NN.parameters(), lr=learning_rate)

optimizer_int = torch.optim.Adam(NN_int.parameters(), lr=learning_rate)

scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optimizer, decay_rate ** (1 / decay_steps)
)

scheduler_int = torch.optim.lr_scheduler.ExponentialLR(
    optimizer_int, decay_rate ** (1 / decay_steps)
)

# ---------------------- FEM Parameters ----------------------#

k_ref = 3

q = 1
k_int = 2
k_test = 1

mesh_sk_H = skfem.MeshTri1().refined(k_ref)

mesh_sk_h = mesh_sk_H.refined(k_test)

V_H = Basis(
    MeshTri(torch.tensor(mesh_sk_H.p).T, torch.tensor(mesh_sk_H.t).T),
    ElementTri(P_order=k_int, int_order=q),
)

V_h = Basis(
    MeshTri(torch.tensor(mesh_sk_h.p).T, torch.tensor(mesh_sk_h.t).T),
    ElementTri(P_order=k_test, int_order=q),
)

I_H, I_H_grad = V_H.interpolate(V_h)

# ---------------------- Residual Parameters ----------------------#

NN_grad_func = lambda x, y: NN_gradiant(NN, x, y)

I_H_NN = lambda x, y: I_H(NN)

I_H_NN_grad = lambda x, y: I_H_grad(NN)

rhs = lambda x, y: 2.0 * math.pi**2 * torch.sin(math.pi * x) * torch.sin(math.pi * y)


def residual(basis, NN_gradient):

    x, y = basis.integration_points

    NN_grad = NN_gradient(x, y)

    v = basis.v
    v_grad = basis.v_grad
    rhs_value = rhs(x, y)

    return rhs_value * v - (v_grad @ NN_grad.mT)


# def gram_matrix(elements: Elements):

#     v = elements.v
#     v_grad = elements.v_grad

#     return v_grad @ v_grad.mT + v @ v.mT

# A = V_h.integrate_bilineal_form(gram_matrix)[V_h.inner_dofs, :][:, V_h.inner_dofs]

# A_inv = torch.linalg.inv(A)

# ---------------------- Error Parameters ----------------------#

exact = lambda x, y: torch.sin(math.pi * x) * torch.sin(math.pi * y)
exact_dx = lambda x, y: math.pi * torch.cos(math.pi * x) * torch.sin(math.pi * y)
exact_dy = lambda x, y: math.pi * torch.sin(math.pi * x) * torch.cos(math.pi * y)


def H1_exact(basis):

    x, y = basis.integration_points

    return exact_dx(x, y) ** 2 + exact_dy(x, y) ** 2 + exact(x, y) ** 2


def H1_norm(basis):

    x, y = basis.integration_points

    NN_dx, NN_dy = torch.split(NN_gradiant(NN, x, y), 1, dim=-1)

    L2_error = (exact(x, y) - NN(x, y)) ** 2

    H1_0_error = (exact_dx(x, y) - NN_dx) ** 2 + (exact_dy(x, y) - NN_dy) ** 2

    return L2_error + H1_0_error


exact_norm = torch.sqrt(torch.sum(V_h.integrate_functional(H1_exact)))

H_1_error_int_list = []
H1_error_list = []

# ---------------------- Training ----------------------#

start_time = datetime.now()

for epoch in range(epochs):
    current_time = datetime.now().strftime("%H:%M:%S")
    print(f"{'='*20} [{current_time}] Epoch:{epoch + 1}/{epochs} {'='*20}")

    residual_value = V_h.integrate_linear_form(residual, NN_grad_func)[V_h.inner_dofs]

    # loss_value = residual_value.T @ (A_inv @ residual_value)

    loss_value = (residual_value**2).sum()

    optimizer_step(optimizer, scheduler, loss_value)

    error_norm = (
        torch.sqrt(torch.sum(V_h.integrate_functional(H1_norm, NN, NN_grad_func)))
        / exact_norm
    )

    residual_value_int = V_h.integrate_linear_form(residual, I_H_NN_grad)[
        V_h.inner_dofs
    ]

    # loss_value_int = residual_value_int.T @ (A_inv @ residual_value_int)

    loss_value_int = (residual_value_int**2).sum()

    optimizer_step(optimizer_int, scheduler_int, loss_value_int)

    error_norm_int = (
        torch.sqrt(torch.sum(V_h.integrate_functional(H1_norm, I_H_NN, I_H_NN_grad)))
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

N_points = 100

x = torch.linspace(0, 1, N_points)
y = torch.linspace(0, 1, N_points)
X, Y = torch.meshgrid(x, y, indexing="ij")

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
