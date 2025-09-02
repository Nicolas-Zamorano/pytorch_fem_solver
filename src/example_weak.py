"# Example of solving a Poisson equation using a neural network and FEM basis functions."

import math
from datetime import datetime
import matplotlib.pyplot as plt
import tensordict as td
import torch
import triangle as tr
from fem import Basis, ElementTri, MeshTri
from neural_network import NeuralNetwork

# torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
# torch.cuda.empty_cache()
torch.set_default_dtype(torch.float64)

# ---------------------- Neural Network Functions ----------------------#


def nn_gradient(neural_net, x, y):
    """Compute the gradient of the neural network output with respect to its inputs."""
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


def optimizer_step(opt, value_loss):
    """Perform an optimization step."""
    opt.zero_grad()
    value_loss.backward(retain_graph=True)
    opt.step()
    scheduler.step()


# ---------------------- Neural Network Parameters ----------------------#

EPOCHS = 5000
LEARNING_RATE = 0.1e-2
DECAY_RATE = 0.99
DECAY_STEPS = 100

NN = torch.jit.script(
    NeuralNetwork(
        input_dimension=2, output_dimension=1, deep_layers=4, hidden_layers_dimension=25
    )
)

optimizer = torch.optim.Adam(NN.parameters(), lr=LEARNING_RATE)

scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optimizer, DECAY_RATE ** (1 / DECAY_STEPS)
)


def nn_grad_func(x, y):
    """Function to compute the gradient of the neural network."""
    return nn_gradient(NN, x, y)


# ---------------------- FEM Parameters ----------------------#

MESH_SIZE = 0.5**8

mesh_data = tr.triangulate(
    {"vertices": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]},
    "Dqea" + str(MESH_SIZE),
)

mesh_data_torch = td.TensorDict(mesh_data)

mesh = MeshTri(triangulation=mesh_data_torch)

elements = ElementTri(polynomial_order=1, integration_order=4)

V = Basis(mesh, elements)

fig, ax_mesh = plt.subplots()

# tr.plot(ax_mesh,**mesh_data)

# ---------------------- Residual Parameters ----------------------#


def rhs(x, y):
    """Right-hand side function."""
    return 2.0 * math.pi**2 * torch.sin(math.pi * x) * torch.sin(math.pi * y)


def residual(basis, gradient):
    """Residual of the PDE."""
    x, y = basis.integration_points

    grad = gradient(x, y)

    v = basis.v
    v_grad = basis.v_grad
    rhs_value = rhs(x, y)

    return rhs_value * v - (v_grad @ grad.mT)


def gram_matrix(basis):
    """Gram matrix of the basis functions."""
    v = basis.v
    v_grad = basis.v_grad

    return v_grad @ v_grad.mT + v @ v.mT


A = V.reduce(V.integrate_bilinear_form(gram_matrix))

A_inv = torch.inverse(A)

# ---------------------- Error Parameters ----------------------#


def exact(x, y):
    """Exact solution of the PDE."""
    return torch.sin(math.pi * x) * torch.sin(math.pi * y)


def exact_dx(x, y):
    """Exact solution derivative with respect to x."""
    return math.pi * torch.cos(math.pi * x) * torch.sin(math.pi * y)


def exact_dy(x, y):
    """Exact solution derivative with respect to y."""
    return math.pi * torch.sin(math.pi * x) * torch.cos(math.pi * y)


def h1_exact(basis):
    """H1 norm of the exact solution."""
    x, y = basis.integration_points

    return exact(x, y) ** 2 + exact_dx(x, y) ** 2 + exact_dy(x, y) ** 2


def h1_norm(basis):
    """H1 norm of the neural network solution."""
    x, y = basis.integration_points

    nn_dx, nn_dy = torch.split(nn_gradient(NN, x, y), 1, dim=-1)

    return (
        (exact(x, y) - NN(x, y)) ** 2
        + (exact_dx(x, y) - nn_dx) ** 2
        + (exact_dy(x, y) - nn_dy) ** 2
    )


exact_norm = torch.sqrt(torch.sum(V.integrate_functional(h1_exact)))

loss_list = []
relative_loss_list = []
H1_error_list = []

LOSS_OPT = 10e4
PARAMS_OPT = NN.state_dict()

# ---------------------- Training ----------------------#

start_time = datetime.now()

for epoch in range(EPOCHS):
    current_time = datetime.now().strftime("%H:%M:%S")
    print(f"{'='*20} [{current_time}] Epoch:{epoch + 1}/{EPOCHS} {'='*20}")

    residual_value = V.reduce(V.integrate_linear_form(residual, nn_grad_func))

    loss_value = residual_value.T @ (A_inv @ residual_value)

    # loss_value = (residual_value**2).sum()

    optimizer_step(optimizer, loss_value)

    error_norm = torch.sqrt(torch.sum(V.integrate_functional(h1_norm))) / exact_norm

    relative_loss = torch.sqrt(loss_value) / exact_norm

    print(
        f"Loss: {loss_value.item():.8f} Relative Loss: {relative_loss.item():.8f} Relative error: {error_norm.item():.8f}"
    )

    if loss_value < LOSS_OPT:
        LOSS_OPT = loss_value
        PARAMS_OPT = NN.state_dict()

    loss_list.append(loss_value.item())
    relative_loss_list.append(relative_loss.item())
    H1_error_list.append(error_norm.item())

end_time = datetime.now()

execution_time = end_time - start_time

print(f"Training time: {execution_time}")

# ---------------------- Plotting ----------------------#

NN.load_state_dict(PARAMS_OPT)

NB_PLOT_POINTS = 100

X, Y = torch.meshgrid(
    torch.linspace(0, 1, NB_PLOT_POINTS),
    torch.linspace(0, 1, NB_PLOT_POINTS),
    indexing="ij",
)

with torch.no_grad():
    Z = abs(exact(X, Y) - NN(X, Y))

figure_solution, axis_solution = plt.subplots()

fig, ax = plt.subplots(dpi=500)
c = ax.contourf(X.cpu(), Y.cpu(), Z.cpu(), levels=100, cmap="viridis")
fig.colorbar(c, ax=ax, orientation="vertical")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title(r"$|u-u_\theta|$")
plt.tight_layout()

figure_error, axis_error = plt.subplots(dpi=500)

axis_error.semilogy(
    relative_loss_list,
    label=r"$\frac{\sqrt{\mathcal{L}(u_\theta)}}{\|u\|_{U}}$",
    linestyle="-.",
)

axis_error.semilogy(
    H1_error_list, label=r"$\frac{\|u-u_\theta\|_{U}}{\|u\|_{U}}$", linestyle=":"
)

axis_error.legend(fontsize=15)

figure_loglog, axis_loglog = plt.subplots()

axis_loglog.loglog(relative_loss_list, H1_error_list)

axis_loglog.set(
    title="Error vs Loss comparison of RVPINNs method",
    xlabel="Relative Loss",
    ylabel="Relative Error",
)

plt.show()
