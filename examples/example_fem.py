import os
from math import pi
import matplotlib.pyplot as plt
import numpy as np
import torch
import triangle as tr
from matplotlib.collections import PolyCollection

from torch_fem import Basis, MeshTri, ElementTri

# torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
# torch.cuda.empty_cache()
torch.set_default_dtype(torch.float64)


INITIAL_VALUE = 0.5
EXPONENT = 10

elements = ElementTri(polynomial_order=1, integration_order=2)


# ---------------------- Residual Parameters ----------------------#

EXPONENTIAL_COEFFICIENT = 5
SCALING_CONSTANT = 1


def bilinear_form(basis: Basis) -> torch.Tensor:
    """Bilinear form."""
    return basis.v_grad @ basis.v_grad.mT


def linear_form(basis: Basis, rhs_values: torch.Tensor) -> torch.Tensor:
    """Linear form for the right-hand side."""
    return rhs_values * basis.v


# ---------------------- Error Parameters ----------------------#


def exact(coordinates: torch.Tensor) -> torch.Tensor:
    x, y = torch.split(coordinates, 1, -1)
    return torch.sin(pi * x) * torch.sin(pi * y)


def exact_dx(coordinates: torch.Tensor) -> torch.Tensor:
    x, y = torch.split(coordinates, 1, -1)
    return pi * torch.cos(pi * x) * torch.sin(pi * y)


def exact_dy(coordinates: torch.Tensor) -> torch.Tensor:
    x, y = torch.split(coordinates, 1, -1)
    return pi * torch.sin(pi * x) * torch.cos(pi * y)


def rhs(coordinates: torch.Tensor) -> torch.Tensor:
    x, y = torch.split(coordinates, 1, -1)
    return 2 * pi**2 * torch.sin(pi * x) * torch.sin(pi * y)


def h1_norm(
    _,
    value: torch.Tensor,
    value_dx: torch.Tensor,
    value_dy: torch.Tensor,
) -> torch.Tensor:
    """H1 norm."""

    return value**2 + value_dx**2 + value_dy**2


mesh_data = tr.triangulate(
    {"vertices": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]},
    "Dqena" + str(INITIAL_VALUE**EXPONENT),
)

mesh = MeshTri(triangulation=mesh_data)
discrete_basis = Basis(mesh, elements)

h_T = discrete_basis.mesh["cells", "length"]

# ---------------------- Recompute precomputed values ----------------------#

integration_points = discrete_basis.integration_points

rhs_value = rhs(integration_points)
exact_value = exact(integration_points)
exact_dx_value = exact_dx(integration_points)
exact_dy_value = exact_dy(integration_points)
exact_norm = torch.sqrt(
    torch.sum(
        discrete_basis.integrate_functional(
            h1_norm, exact_value, exact_dx_value, exact_dy_value
        )
    )
)

b = discrete_basis.integrate_linear_form(linear_form, rhs_value)
A = discrete_basis.integrate_bilinear_form(bilinear_form)
solution = discrete_basis.solve(A, b)

interpolated_solution, interpolated_solution_grad = discrete_basis.interpolate(
    discrete_basis, solution
)

interpolated_solution_dx, interpolated_solution_dy = torch.split(
    interpolated_solution_grad, 1, -1
)

h1_error_plot = (
    torch.sqrt(
        discrete_basis.integrate_functional(
            h1_norm,
            interpolated_solution - exact_value,
            interpolated_solution_dx - exact_dx_value,
            interpolated_solution_dy - exact_dy_value,
        )
    )
    .squeeze(-1)
    .numpy(force=True)
)

# ---------------------- Save plots and data ----------------------#

# Solution plot
figure_solution, axis_solution = plt.subplots()
c4e = torch.Tensor.numpy(discrete_basis.mesh["cells", "coordinates"], force=True)

triangles_plot = PolyCollection(
    c4e,  # type: ignore
    array=h1_error_plot,
    cmap="viridis",
    edgecolors="black",
    linewidths=0.2,
)

axis_solution.add_collection(triangles_plot)
axis_solution.autoscale_view()
axis_solution.set_xlabel("x")
axis_solution.set_ylabel("y")
axis_solution.set_xlim((0, 1))
axis_solution.set_ylim((0, 1))
color_bar = plt.colorbar(triangles_plot, ax=axis_solution)
color_bar.set_label(r"$H^1$ error")
figure_solution.tight_layout()

plt.show()
