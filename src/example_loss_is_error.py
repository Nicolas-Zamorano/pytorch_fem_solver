"""Example of using Example solution to train NN to to solve a Poisson
equation with Dirichlet boundary conditions."""

import math
from datetime import datetime

import matplotlib.pyplot as plt
import torch
from Triangulation import Triangulation

from fem import Basis, ElementTri, MeshTri
from neural_network import NeuralNetwork

torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
torch.set_default_dtype(torch.float64)
torch.manual_seed(1234)

# ---------------------- Neural Network Parameters ----------------------#


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


def nn_laplacian(neural_net, x, y):
    """
    Compute Laplacian of a Neural Network w.r.t inputs.

    Parameters:
    - inputs (torch.Tensor): input tensors.

    Returns:
    - laplacian (torch.Tensor): The Laplacian of the network output w.r.t each input tensor.
    """

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

    du_xx = torch.autograd.grad(
        outputs=gradients[0],
        inputs=(x, y),
        grad_outputs=torch.ones_like(gradients[0]),
        retain_graph=True,
        create_graph=True,
    )[0]

    du_yy = torch.autograd.grad(
        outputs=gradients[1],
        inputs=(x, y),
        grad_outputs=torch.ones_like(gradients[1]),
        retain_graph=True,
        create_graph=True,
    )[1]

    lap = du_xx + du_yy

    return lap


def optimizer_step(opt, value_loss):
    """Perform an optimization step."""
    opt.zero_grad()
    value_loss.backward(retain_graph=True)
    opt.step()
    scheduler.step()


EPOCHS = 1000
LEARNING_RATE = 0.5e-1
DECAY_RATE = 0.95
DECAY_STEPS = 100

NN = torch.jit.script(
    NeuralNetwork(
        input_dimension=2, output_dimension=1, deep_layers=5, hidden_layers_dimension=10
    )
)

optimizer = torch.optim.Adam(NN.parameters(), lr=0.1e-2)

scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optimizer, DECAY_RATE ** (1 / DECAY_STEPS)
)

# ---------------------- FEM Parameters ----------------------#

c4n, n4e = Triangulation(3)

coords4nodes = torch.tensor(c4n)

nodes4elements = torch.tensor(n4e)

mesh = MeshTri(coords4nodes, nodes4elements)

mesh = MeshTri(coords4nodes, nodes4elements)

elements = ElementTri(polynomial_order=1, integration_order=2)

V = Basis(mesh, elements)

# ---------------------- Residual Parameters ----------------------#


def rhs(x, y):
    """Right-hand side function."""
    return -2 * math.pi**2 * torch.sin(math.pi * x) * torch.sin(math.pi * y)


def residual(basis):
    """Compute the residual of the Poisson equation using the neural network as an approximation."""
    x, y = basis.integration_points

    # NN_grad = NN_gradient(NN, x, y)
    v = basis.v
    # v_grad = basis.v_grad
    rhs_value = rhs(x, y)
    nn_lap = nn_laplacian(NN, x, y)

    return (nn_lap + rhs_value(x, y)) * v

    # return NN_grad @ v_grad.mT - rhs_value * v


def gram_matrix(basis):
    """Compute the Gram matrix for the basis functions."""
    v = basis.v
    v_grad = basis.v_grad

    return 3 * v_grad @ v_grad.mT + v @ v.mT
    # return 3 * v_grad @ v_grad.mT
    # return v @ v.mT


A = V.integrate_bilinear_form(gram_matrix)

A_inv = torch.inverse(V.integrate_bilinear_form(gram_matrix))

# ---------------------- Error Parameters ----------------------#


def h1_exact(basis):
    """Compute the H1 norm of the exact solution."""
    x, y = basis.integration_points

    exact = (torch.sin(math.pi * x) * torch.sin(math.pi * y)) / (2 * math.pi**2)

    exact_dx = (torch.cos(math.pi * x) * torch.sin(math.pi * y)) / (2 * math.pi)
    exact_dy = (torch.sin(math.pi * x) * torch.cos(math.pi * y)) / (2 * math.pi)

    return (exact_dx**2 + exact_dy**2) ** 2 + exact**2


def h1_norm(basis):
    """Compute the H1 norm of the neural network solution."""
    x, y = basis.integration_points

    nn_dx, nn_dy = torch.split(nn_gradient(NN, x, y), 1, dim=-1)

    exact = (torch.sin(math.pi * x) * torch.sin(math.pi * y)) / (2 * math.pi**2)

    exact_dx = (torch.cos(math.pi * x) * torch.sin(math.pi * y)) / (2 * math.pi)
    exact_dy = (torch.sin(math.pi * x) * torch.cos(math.pi * y)) / (2 * math.pi)

    l2_error = (exact - NN(x, y)) ** 2

    h1_0_error = (exact_dx - nn_dx) ** 2 + (exact_dy - nn_dy) ** 2

    return l2_error + h1_0_error


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

    # residual_value = V.integrate_lineal_form(residual)

    # loss_value = residual_value.T @ (A_inv @ residual_value)

    loss_value = torch.sqrt(torch.sum(V.integrate_functional(h1_norm)))

    optimizer_step(optimizer, loss_value)

    error_norm = torch.sqrt(torch.sum(V.integrate_functional(h1_norm))) / exact_norm

    # relative_loss = torch.sqrt(loss_value)/exact_norm

    relative_loss = loss_value / exact_norm

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
    Z = NN(X, Y)

figure_solution = plt.figure()
axis_solution = figure_solution.add_subplot(111, projection="3d")

contour = axis_solution.plot_surface(
    X.cpu().detach().numpy(),
    Y.cpu().detach().numpy(),
    Z.cpu().detach().numpy(),
    cmap="viridis",
)

axis_solution.set(
    title="Solution obtain with RVPINNs method",
    xlabel="x",
    ylabel="y",
    zlabel=r"$u_\theta(x,y)$",
)

figure_loss, axis_loss = plt.subplots()

axis_loss.semilogy(loss_list)

axis_loss.set(
    title="Error evolution of RVPINNs method", xlabel="# epochs", ylabel="Error"
)

figure_error, axis_error = plt.subplots()

axis_error.semilogy(
    relative_loss_list,
    label=r"$\frac{\sqrt{\mathcal{L}(u_\theta)}}{\|u\|_{V}}$",
    linestyle="-.",
)

axis_error.semilogy(
    H1_error_list, label=r"$\frac{\|u-u_\theta\|_{V}}{\|u\|_{V}}$", linestyle=":"
)

axis_error.legend()

figure_loglog, axis_loglog = plt.subplots()

axis_loglog.loglog(relative_loss_list, H1_error_list)

axis_loglog.set(
    title="Error vs Loss comparison of RVPINNs method",
    xlabel="Relative Loss",
    ylabel="Relative Error",
)


plt.show()
