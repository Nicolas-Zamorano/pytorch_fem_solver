"# Example of solving a Poisson equation using FEM."

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import torch
import triangle as tr

from torch_fem import (
    Basis,
    ElementTri,
    MeshTri,
)

torch.set_default_dtype(torch.float64)

mesh_data = tr.triangulate(
    {"vertices": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]},
    "Dqena" + str(0.5**8),
)

mesh = MeshTri(triangulation=mesh_data)

elements = ElementTri(polynomial_order=1, integration_order=10)

discrete_basis = Basis(mesh, elements)

EXPONENTIAL_COEFFICIENT = 1
SCALING_CONSTANT = 20


def rhs(coordinates: torch.Tensor) -> torch.Tensor:
    """Right-hand side function."""
    x, y = torch.split(coordinates, 1, -1)

    exponential_value = torch.exp(EXPONENTIAL_COEFFICIENT * x)

    gxx = (
        -2 * (exponential_value - 1)
        + 2 * EXPONENTIAL_COEFFICIENT * (1 - 2 * x) * exponential_value
        + EXPONENTIAL_COEFFICIENT**2 * x * (1 - x) * exponential_value
    )

    fxx = SCALING_CONSTANT * y * (1 - y) * gxx

    fyy = SCALING_CONSTANT * (-2) * x * (1 - x) * (exponential_value - 1)

    lap = fxx + fyy
    return -lap


def gram_matrix(basis: Basis) -> torch.Tensor:
    """Gram matrix of the basis functions."""
    return basis.v_grad @ basis.v_grad.mT


def linear_form(basis: Basis) -> torch.Tensor:
    """rhs linear form"""
    return rhs(basis.integration_points) * basis.v


A = discrete_basis.integrate_bilinear_form(gram_matrix)
b = discrete_basis.integrate_linear_form(linear_form)

solution = discrete_basis.solution_tensor()

discrete_basis.solve(A, solution, b)

figure_solution, axis_solution = plt.subplots(subplot_kw={"projection": "3d"})  # type: ignore

vertices_plot = discrete_basis._coords4global_dofs.numpy(force=True)
triangles_plot = discrete_basis._global_dofs4elements.numpy(force=True)
solution_plot = solution.squeeze(-1).numpy(force=True)

axis_solution.plot_trisurf(
    vertices_plot[..., 0],
    vertices_plot[..., 1],
    solution_plot,
    triangles=triangles_plot,
    cmap="viridis",
    edgecolor="black",
    linewidth=0.1,
)

plt.show()
