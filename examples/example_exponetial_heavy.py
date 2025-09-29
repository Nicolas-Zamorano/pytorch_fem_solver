"# Example of solving a Poisson equation using a neural network and FEM basis functions."

import math

import matplotlib.pyplot as plt
import torch
import triangle as tr

from torch_fem import (
    Basis,
    ElementTri,
    MeshTri,
    ElementLine,
    InteriorEdgesBasis,
    FeedForwardNeuralNetwork as NeuralNetwork,
    Model,
)


# torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
# torch.cuda.empty_cache()
torch.set_default_dtype(torch.float64)


# ---------------------- Neural Network Parameters ----------------------#


class BoundaryConstrain(torch.nn.Module):
    """Class to strongly apply bc"""

    def forward(self, inputs):
        """Boundary condition modifier function."""
        x, y = torch.split(inputs, 1, dim=-1)
        return x * (x - 1) * y * (y - 1)


NN = NeuralNetwork(
    input_dimension=2,
    output_dimension=1,
    nb_hidden_layers=4,
    neurons_per_layers=40,
    boundary_condition_modifier=BoundaryConstrain(),
    use_xavier_initialization=False,
)

# ---------------------- FEM Parameters ----------------------#

mesh_data = tr.triangulate(
    {"vertices": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]},
    "Dqena" + str(0.5**5),
)

mesh = MeshTri(triangulation=mesh_data)

elements = ElementTri(polynomial_order=1, integration_order=2)

discrete_basis = Basis(mesh, elements)

elements_1D = ElementLine(polynomial_order=1, integration_order=2)

V_edges = InteriorEdgesBasis(mesh, elements_1D)

_, interpolator_to_edges_grad = discrete_basis.interpolate(V_edges)

h_T = discrete_basis.mesh["cells", "length"]
h_E = discrete_basis.mesh["interior_edges", "length"].unsqueeze(-2)
n_E = discrete_basis.mesh["interior_edges", "normals"].unsqueeze(-2)

# ---------------------- Residual Parameters ----------------------#


def rhs(coordinates):
    """Right-hand side function."""
    x, y = torch.split(coordinates, 1, -1)

    a = x * (1 - x)
    a_p = 1 - 2 * x
    A_pp = -2
    a = y * (1 - y)
    b_p = 1 - 2 * y
    B_pp = -2
    a = torch.exp(-80 * (1 - x**2 - y**2))
    phi = a - 1
    phi_x = 160 * x * a
    phi_y = 160 * y * a
    phi_xx = a * (160 + 25600 * x**2)
    phi_yy = a * (160 + 25600 * y**2)

    u_xx = 1e-2 * a * (A_pp * phi + 2 * a_p * phi_x + a * phi_xx)
    u_yy = 1e-2 * a * (B_pp * phi + 2 * b_p * phi_y + a * phi_yy)
    return -(u_xx + u_yy)


def residual(basis, neural_network):
    """Residual of the PDE."""
    integration_points = basis.integration_points

    grad = neural_network.gradient(integration_points)

    v = basis.v
    v_grad = basis.v_grad
    rhs_value = rhs(integration_points)

    return rhs_value * v - (v_grad @ grad.mT)


def gram_matrix(basis):
    """Gram matrix of the basis functions."""
    v_grad = basis.v_grad

    return v_grad @ v_grad.mT


def jump(_, normal_elements, edge_size, neural_network):
    """Jump term for discontinuous solutions"""
    interpolator_u_grad_plus, interpolator_u_grad_minus = torch.unbind(
        interpolator_to_edges_grad(neural_network), dim=-4
    )
    return (
        edge_size
        * (
            (interpolator_u_grad_plus * normal_elements).sum(-1, keepdim=True)
            + (interpolator_u_grad_minus * -normal_elements).sum(-1, keepdim=True)
        )
        ** 2
    )


def rhs_term(basis, triangle_size, neural_network):
    """Residual term for the right-hand side"""
    integration_points = basis.integration_points

    return triangle_size**2 * (
        rhs(integration_points) + neural_network.laplacian(integration_points)
    )


gram_matrix_inverse = torch.inverse(
    discrete_basis.reduce(discrete_basis.integrate_bilinear_form(gram_matrix))
)


# ---------------------- Error Parameters ----------------------#


def exact(coordinates):
    """Exact solution of the PDE."""
    x, y = torch.split(coordinates, 1, -1)

    a = x * (1 - x)
    a = y * (1 - y)
    a = torch.exp(-80 * (1 - x**2 - y**2))
    return 1e-2 * a * a * (a - 1)


def exact_dx(coordinates):
    """Exact solution derivative with respect to x."""
    x, y = torch.split(coordinates, 1, -1)

    a = x * (1 - x)
    a_p = 1 - 2 * x
    a = y * (1 - y)
    a = torch.exp(-80 * (1 - x**2 - y**2))
    phi = a - 1
    phi_x = 160 * x * a
    return 1e-2 * a * (a_p * phi + a * phi_x)


def exact_dy(coordinates):
    """Exact solution derivative with respect to y."""
    x, y = torch.split(coordinates, 1, -1)

    a = x * (1 - x)
    a = y * (1 - y)
    b_p = 1 - 2 * y
    a = torch.exp(-80 * (1 - x**2 - y**2))
    phi = a - 1
    phi_y = 160 * y * a
    return 1e-2 * a * (b_p * phi + a * phi_y)


def h1_exact(basis):
    """H1 norm of the exact solution."""
    integration_points = basis.integration_points

    return (
        exact(integration_points) ** 2
        + exact_dx(integration_points) ** 2
        + exact_dy(integration_points) ** 2
    )


def h1_norm(basis, neural_network, gradient):
    """H1 norm of the neural network solution."""
    integration_points = basis.integration_points

    nn_dx, nn_dy = torch.split(gradient(integration_points), 1, dim=-1)

    return (
        (exact(integration_points) - neural_network(integration_points)) ** 2
        + (exact_dx(integration_points) - nn_dx) ** 2
        + (exact_dy(integration_points) - nn_dy) ** 2
    )


exact_norm = torch.sqrt(torch.sum(discrete_basis.integrate_functional(h1_exact)))

# ---------------------- Training ----------------------#


def training_step(neural_network):
    """Training step for the neural network."""
    residual_vector = discrete_basis.reduce(
        discrete_basis.integrate_linear_form(residual, neural_network)
    )

    loss_value = residual_vector.T @ (gram_matrix_inverse @ residual_vector)

    posteriori = (
        discrete_basis.integrate_functional(rhs_term, h_T, neural_network) ** 2
    ).sum() + (V_edges.integrate_functional(jump, n_E, h_E, neural_network) ** 2).sum()

    # loss_value = torch.sum(residual_vector**2) + posteriori

    loss_value += posteriori

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
    epochs=8000,
    optimizer=torch.optim.Adam,
    optimizer_kwargs={"lr": 0.001},
    # learning_rate_scheduler=torch.optim.lr_scheduler.ExponentialLR,
    # scheduler_kwargs={"gamma": 0.99**100},
    use_early_stopping=True,
    early_stopping_patience=50,
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
    Z = abs(exact(plot_points) - NN(plot_points)).squeeze(-1)

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
        "title": "RVPINNs",
    }
)

plt.show()
