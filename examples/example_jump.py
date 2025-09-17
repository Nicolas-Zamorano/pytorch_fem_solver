"""Example of using jump residual for a problem with discontinuous solution"""

import math

import triangle as tr
import matplotlib.pyplot as plt
import torch

from fem import Basis, ElementLine, ElementTri, InteriorEdgesBasis, MeshTri
from model import Model, FeedForwardNeuralNetwork as NeuralNetwork

# torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
# torch.cuda.empty_cache()
torch.set_default_dtype(torch.float64)


def boundary_constraint(x):
    """Boundary condition modifier function."""
    return x[..., [0]] * (x[..., [0]] - 1) * x[..., [1]] * (x[..., [1]] - 1)


NN = NeuralNetwork(
    input_dimension=2,
    output_dimension=1,
    nb_hidden_layers=4,
    neurons_per_layers=25,
    boundary_condition_modifier=boundary_constraint,
)
# ---------------------- FEM Parameters ----------------------#

mesh_data = tr.triangulate(
    {"vertices": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]},
    "Dqena" + str(0.5**8),
)

mesh = MeshTri(triangulation=mesh_data)

elements = ElementTri(polynomial_order=1, integration_order=2)

elements_1D = ElementLine(polynomial_order=1, integration_order=2)

V_edges = InteriorEdgesBasis(mesh, elements_1D)

V = Basis(mesh, elements)

_, interpolator_to_edges_grad = V.interpolate(V_edges)

_, interpolators_grad = V.interpolate(V)

# ---------------------- Residual Parameters ----------------------#


def rhs(x, y):
    """Right-hand side function"""
    return 2.0 * math.pi**2 * torch.sin(math.pi * x) * torch.sin(math.pi * y)


h_T = V.mesh["cells"]["length"]
h_E = V.mesh["interior_edges"]["length"].unsqueeze(-2)
n_E = V.mesh["interior_edges"]["normals"].unsqueeze(-2)


def jump(_, normal_elements, edge_size, nn):
    """Jump term for discontinuous solutions"""
    interpolator_u_grad_plus, interpolator_u_grad_minus = torch.unbind(
        interpolator_to_edges_grad(nn), dim=-4
    )
    return (
        edge_size
        * (
            (interpolator_u_grad_plus * normal_elements).sum(-1, keepdim=True)
            + (interpolator_u_grad_minus * -normal_elements).sum(-1, keepdim=True)
        )
        ** 2
    )


def rhs_term(basis, triangle_size, nn):
    """Residual term for the right-hand side"""
    x, y = torch.split(basis.integration_points, 1, dim=-1)

    return triangle_size**2 * (rhs(x, y) + interpolators_grad(nn))


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
    x, y = torch.split(basis.integration_points, 1, dim=-1)

    return exact(x, y) ** 2 + exact_dx(x, y) ** 2 + exact_dy(x, y) ** 2


def h1_norm(basis, neural_network, gradient):
    """H1 norm of the neural network solution."""
    integration_points = basis.integration_points
    x, y = torch.split(integration_points, 1, dim=-1)

    nn_dx, nn_dy = torch.split(gradient(integration_points), 1, dim=-1)

    return (
        (exact(x, y) - neural_network(integration_points)) ** 2
        + (exact_dx(x, y) - nn_dx) ** 2
        + (exact_dy(x, y) - nn_dy) ** 2
    )


exact_norm = torch.sqrt(torch.sum(V.integrate_functional(h1_exact)))

# ---------------------- Training ----------------------#


def training_step(neural_network):
    """Training step for the neural network."""
    loss_value = (V.integrate_functional(rhs_term, h_T, neural_network) ** 2).sum() + (
        V_edges.integrate_functional(jump, n_E, h_E, neural_network) ** 2
    ).sum()

    relative_loss = torch.sqrt(loss_value) / exact_norm**2

    h1_error = torch.sqrt(
        torch.sum(
            V.integrate_functional(h1_norm, neural_network, neural_network.gradient)
        )
    )

    return loss_value, relative_loss, h1_error / exact_norm


model = Model(
    neural_network=NN,
    training_step=training_step,
    epochs=15000,
    optimizer=torch.optim.Adam,
    optimizer_kwargs={"lr": 0.001},
    # learning_rate_scheduler=torch.optim.lr_scheduler.ExponentialLR,
    # scheduler_kwargs={"gamma": 0.9},
    use_early_stopping=False,
    early_stopping_patience=100,
    min_delta=1e-12,
)

model.train()

# ---------------------- Plotting ----------------------#

model.load_optimal_parameters()

linspace = torch.linspace(0, 1, 100)

X, Y = torch.stack(
    torch.meshgrid(
        linspace,
        linspace,
        indexing="ij",
    )
)

plot_points = torch.stack([X, Y], dim=-1)

with torch.no_grad():
    Z = abs(exact(X, Y) - NN(plot_points).squeeze(-1))

figure_solution, axis_solution = plt.subplots()
contour_solution = axis_solution.contourf(
    X.numpy(force=True),
    Y.numpy(force=True),
    Z.numpy(force=True),
    levels=100,
    cmap="viridis",
)
figure_solution.colorbar(contour_solution, ax=axis_solution, orientation="vertical")

axis_solution.set_xlabel("x")
axis_solution.set_ylabel("y")
axis_solution.set_title(r"$|u-u_\theta|$")
plt.tight_layout()

model.plot_training_history(
    plot_names={
        "loss": r"$\mathcal{L}(u_{\theta})$",
        "validation": r"$\frac{\sqrt{\mathcal{L}(u_{\theta})}}{\|u\|_U}$",
        "accuracy": r"$\frac{\|u-u_{\theta}\|_U}{\|u_{\theta}\|_U}$",
        "title": "only a posteriori estimator",
    }
)

plt.show()
