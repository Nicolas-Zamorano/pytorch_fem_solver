import torch
import math

import matplotlib.pyplot as plt

from Neural_Network import Neural_Network
from fem import Mesh, Elements, Basis
from datetime import datetime
from Triangulation import Triangulation

torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
torch.set_default_dtype(torch.float64)
torch.manual_seed(1234)

# ---------------------- Neural Network Parameters ----------------------#


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


def NN_laplacian(NN, x, y):
    """
    Compute Laplacian of a Neural Network w.r.t inputs.

    Parameters:
    - inputs (torch.Tensor): input tensors.

    Returns:
    - laplacian (torch.Tensor): The Laplacian of the network output w.r.t each input tensor.
    """

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


def optimizer_step(optimizer, loss_value):
    optimizer.zero_grad()
    loss_value.backward(retain_graph=True)
    optimizer.step()
    scheduler.step()


epochs = 1000
learning_rate = 0.5e-1
decay_rate = 0.95
decay_steps = 100

NN = torch.jit.script(
    Neural_Network(
        input_dimension=2, output_dimension=1, deep_layers=5, hidden_layers_dimension=10
    )
)

optimizer = torch.optim.Adam(NN.parameters(), lr=0.1e-2)

scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optimizer, decay_rate ** (1 / decay_steps)
)

# ---------------------- FEM Parameters ----------------------#

c4n, n4e = Triangulation(3)

coords4nodes = torch.tensor(c4n)

nodes4elements = torch.tensor(n4e)

mesh = Mesh(coords4nodes, nodes4elements)

mesh = Mesh(coords4nodes, nodes4elements)

elements = Elements(P_order=1, int_order=2)

V = Basis(mesh, elements)

# ---------------------- Residual Parameters ----------------------#

rhs = lambda x, y: torch.sin(math.pi * x) * torch.sin(math.pi * y)


def residual(elements: Elements):

    x, y = elements.integration_points

    # NN_grad = NN_gradiant(NN, x, y)
    v = elements.v
    v_grad = elements.v_grad
    rhs_value = rhs(x, y)
    NN_lap = NN_laplacian(NN, x, y)

    return (NN_lap + rhs_value(x, y)) * v

    # return NN_grad @ v_grad.mT - rhs_value * v


def gram_matrix(elements: Elements):

    v = elements.v
    v_grad = elements.v_grad

    return 3 * v_grad @ v_grad.mT + v @ v.mT
    # return 3 * v_grad @ v_grad.mT
    # return v @ v.mT


A = V.integrate_bilineal_form(gram_matrix)

A_inv = torch.linalg.inv(V.integrate_bilineal_form(gram_matrix))

# ---------------------- Error Parameters ----------------------#


def H1_exact(elements: Elements):

    x, y = elements.integration_points

    exact = (torch.sin(math.pi * x) * torch.sin(math.pi * y)) / (2 * math.pi**2)

    exact_dx = (torch.cos(math.pi * x) * torch.sin(math.pi * y)) / (2 * math.pi)
    exact_dy = (torch.sin(math.pi * x) * torch.cos(math.pi * y)) / (2 * math.pi)

    return (exact_dx**2 + exact_dy**2) ** 2 + exact**2


def H1_norm(elements: Elements):

    x, y = elements.integration_points

    NN_dx, NN_dy = torch.split(NN_gradiant(NN, x, y), 1, dim=-1)

    exact = (torch.sin(math.pi * x) * torch.sin(math.pi * y)) / (2 * math.pi**2)

    exact_dx = (torch.cos(math.pi * x) * torch.sin(math.pi * y)) / (2 * math.pi)
    exact_dy = (torch.sin(math.pi * x) * torch.cos(math.pi * y)) / (2 * math.pi)

    L2_error = (exact - NN(x, y)) ** 2

    H1_0_error = (exact_dx - NN_dx) ** 2 + (exact_dy - NN_dy) ** 2

    return L2_error + H1_0_error


exact_norm = torch.sqrt(torch.sum(V.integrate_functional(H1_exact)))

loss_list = []
relative_loss_list = []
H1_error_list = []

loss_opt = 10e4

# ---------------------- Training ----------------------#

start_time = datetime.now()

for epoch in range(epochs):
    current_time = datetime.now().strftime("%H:%M:%S")
    print(f"{'='*20} [{current_time}] Epoch:{epoch + 1}/{epochs} {'='*20}")

    # residual_value = V.integrate_lineal_form(residual)

    # loss_value = residual_value.T @ (A_inv @ residual_value)

    loss_value = torch.sqrt(torch.sum(V.integrate_functional(H1_norm)))

    optimizer_step(optimizer, loss_value)

    error_norm = torch.sqrt(torch.sum(V.integrate_functional(H1_norm))) / exact_norm

    # relative_loss = torch.sqrt(loss_value)/exact_norm

    relative_loss = loss_value / exact_norm

    print(
        f"Loss: {loss_value.item():.8f} Relative Loss: {relative_loss.item():.8f} Relative error: {error_norm.item():.8f}"
    )

    if loss_value < loss_opt:
        loss_opt = loss_value
        params_opt = NN.state_dict()

    loss_list.append(loss_value.item())
    relative_loss_list.append(relative_loss.item())
    H1_error_list.append(error_norm.item())

end_time = datetime.now()

execution_time = end_time - start_time

print(f"Training time: {execution_time}")

# ---------------------- Plotting ----------------------#

NN.load_state_dict(params_opt)

N_points = 100

x = torch.linspace(0, 1, N_points)
y = torch.linspace(0, 1, N_points)
X, Y = torch.meshgrid(x, y, indexing="ij")

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
    title="Error vs Loss comparasion of RVPINNs method",
    xlabel="Relative Loss",
    ylabel="Relative Error",
)


plt.show()
