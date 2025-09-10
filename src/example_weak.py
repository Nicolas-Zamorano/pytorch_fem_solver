"# Example of solving a Poisson equation using a neural network and FEM basis functions."

import math

import matplotlib.pyplot as plt
import torch
import triangle as tr

from fem import Basis, ElementTri, MeshTri
from model import FeedForwardNeuralNetwork as NeuralNetwork, Model

# torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
# torch.cuda.empty_cache()
torch.set_default_dtype(torch.float64)


# ---------------------- Neural Network Parameters ----------------------#


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

discrete_basis = Basis(mesh, elements)

# ---------------------- Residual Parameters ----------------------#


def rhs(x, y):
    """Right-hand side function."""
    return 2.0 * math.pi**2 * torch.sin(math.pi * x) * torch.sin(math.pi * y)


def residual(basis, gradient):
    """Residual of the PDE."""
    integration_points = basis.integration_points
    x, y = torch.split(integration_points, 1, dim=-1)

    grad = gradient(integration_points)

    v = basis.v
    v_grad = basis.v_grad
    rhs_value = rhs(x, y)

    return rhs_value * v - (v_grad @ grad.mT)


def gram_matrix(basis):
    """Gram matrix of the basis functions."""
    v_grad = basis.v_grad

    return v_grad @ v_grad.mT


gram_matrix_inverse = torch.inverse(
    discrete_basis.reduce(discrete_basis.integrate_bilinear_form(gram_matrix))
)

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


exact_norm = torch.sqrt(torch.sum(discrete_basis.integrate_functional(h1_exact)))

# ---------------------- Training ----------------------#


def training_step(neural_network):
    """Training step for the neural network."""
    residual_vector = discrete_basis.reduce(
        discrete_basis.integrate_linear_form(residual, neural_network.gradient)
    )

    # loss_value = residual_vector.T @ (gram_matrix_inverse @ residual_vector)

    loss_value = torch.sum(residual_vector**2, dim=0)

    relative_loss = torch.sqrt(loss_value) / exact_norm**2

    h1_error = torch.sqrt(
        torch.sum(
            discrete_basis.integrate_functional(
                h1_norm, neural_network, neural_network.gradient
            )
        )
    )

    return loss_value, relative_loss, h1_error / exact_norm


model = Model(
    neural_network=NN,
    training_step=training_step,
    epochs=5000,
    learning_rate=0.1e-1,
    use_decay_learning_rate=True,
    decay_rate=0.99,
    decay_steps=100,
    use_early_stopping=True,
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

model.plot_training_history()

plt.show()
