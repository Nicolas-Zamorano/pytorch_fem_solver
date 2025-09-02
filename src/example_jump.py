"""Example of using jump residual for a problem with discontinuous solution"""

import math
from datetime import datetime

import matplotlib.pyplot as plt
import skfem
import torch

from fem import Basis, ElementLine, ElementTri, InteriorFacetBasis, MeshTri
from neural_network import NeuralNetwork

torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
torch.set_default_dtype(torch.float64)

# ---------------------- Neural Network Functions ----------------------#


def nn_gradient(neural_net, x, y):
    """compute gradient of a Neural Network w.r.t inputs."""
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

    # return gradients
    return torch.concat(gradients, dim=-1)


def optimizer_step(opt, value_loss):
    """Performs a single optimization step."""
    opt.zero_grad()
    value_loss.backward(retain_graph=True)
    opt.step()
    scheduler.step()


# ---------------------- Neural Network Parameters ----------------------#

EPOCHS = 5000
LEARNING_RATE = 0.5e-2
DECAY_RATE = 0.99
DECAY_STEPS = 100

NN = torch.jit.script(
    NeuralNetwork(
        input_dimension=2, output_dimension=1, deep_layers=4, hidden_layers_dimension=40
    )
)

optimizer = torch.optim.Adam(NN.parameters(), lr=LEARNING_RATE)

scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optimizer, DECAY_RATE ** (1 / DECAY_STEPS)
)

# ---------------------- FEM Parameters ----------------------#

mesh_sk = skfem.MeshTri1().refined(3)

coords4nodes = torch.tensor(mesh_sk.p).T

nodes4elements = torch.tensor(mesh_sk.t).T

mesh = MeshTri(coords4nodes, nodes4elements)

elements = ElementTri(polynomial_order=1, integration_order=2)

elements_1D = ElementLine(polynomial_order=1, integration_order=2)

V_edges = InteriorFacetBasis(mesh, elements_1D)

V = Basis(mesh, elements)

I, grad_interpolator = V.interpolate(V_edges)

# ---------------------- Residual Parameters ----------------------#


def rhs(x, y):
    """Right-hand side function"""
    return 2.0 * math.pi**2 * torch.sin(math.pi * x) * torch.sin(math.pi * y)


normals = mesh.normal4inner_edges.unsqueeze(-2).unsqueeze(-2)
h_E = mesh.inner_edges_length.unsqueeze(-1).unsqueeze(-1)
h_T = mesh.elements_diameter.unsqueeze(-1).unsqueeze(-1)


def jump(_, normal_elements, edge_size):
    """Jump term for discontinuous solutions"""
    interpolator_u_grad_plus, interpolator_u_grad_minus = torch.unbind(
        grad_interpolator(NN), dim=-4
    )
    return (
        edge_size
        * (
            (interpolator_u_grad_plus * normal_elements).sum(-1, keepdim=True)
            + (interpolator_u_grad_minus * -normal_elements).sum(-1, keepdim=True)
        )
        ** 2
    )


def rhs_term(basis, triangle_size):
    """Residual term for the right-hand side"""
    x, y = basis.integration_points

    return triangle_size**2 * rhs(x, y) ** 2


# ---------------------- Error Parameters ----------------------#


def h1_exact(basis):
    """Exact solution for H1 norm computation"""
    x, y = basis.integration_points

    exact = torch.sin(math.pi * x) * torch.sin(math.pi * y)

    exact_dx = math.pi * torch.cos(math.pi * x) * torch.sin(math.pi * y)
    exact_dy = math.pi * torch.sin(math.pi * x) * torch.cos(math.pi * y)

    return exact_dx**2 + exact_dy**2 + exact**2


def h1_norm(basis):
    """H1 norm computation"""
    x, y = basis.integration_points

    nn_dx, nn_dy = torch.split(nn_gradient(NN, x, y), 1, dim=-1)

    exact = torch.sin(math.pi * x) * torch.sin(math.pi * y)

    exact_dx = math.pi * torch.cos(math.pi * x) * torch.sin(math.pi * y)
    exact_dy = math.pi * torch.sin(math.pi * x) * torch.cos(math.pi * y)

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

    residual_value = (V.integrate_functional(rhs_term, h_T)).sum() + (
        V_edges.integrate_functional(jump, normals, h_E)
    ).sum()

    loss_value = residual_value

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
    Z = abs(torch.sin(math.pi * X) * torch.sin(math.pi * Y) - NN(X, Y))

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

# figure_loglog, axis_loglog = plt.subplots()

# axis_loglog.loglog(relative_loss_list,
#                     H1_error_list)

# axis_loglog.set(title = "Error vs Loss comparison of RVPINNs method",
#                 xlabel = "Relative Loss",
#                 ylabel = "Relative Error")

plt.show()
