"""1 fracture 2D test case to compare with fracture implementation."""

import torch

import matplotlib.pyplot as plt
import tensordict as td
import triangle as tr
from ..torch_fem.basis import AbstractBasis
from torch_fem import MeshTri, ElementTri, Basis

torch.set_default_dtype(torch.float64)

# pylint: disable=not-callable


def rhs(x, y):
    """Right-hand side function."""
    return 6 * x * (y - y**2) + 2 * x * (1 - x**2)


def l(basis):
    """Linear form."""
    integration_points = basis.integration_points
    x, y = torch.split(integration_points, 1, dim=-1)

    v = basis.v

    return rhs(x, y) * v


def a(basis):
    """Bilinear form."""
    v_grad = basis.v_grad

    return v_grad @ v_grad.mT


def exact(x, y):
    """Exact solution."""
    return y * (1 - y) * x * (1 - x**2)


def exact_dx(x, y):
    """Derivative of the exact solution with respect to x."""
    return -y * (1 - y) * ((x**2 - 1) + 2 * x * x)


def exact_dy(x, y):
    """Derivative of the exact solution with respect to y."""
    return -(1 - 2 * y) * x * (x**2 - 1)


def h1_exact(basis: AbstractBasis) -> torch.Tensor:
    """H1 norm of the exact solution."""
    integration_points = basis.integration_points
    x, y = torch.split(integration_points, 1, dim=-1)

    return exact(x, y) ** 2 + exact_dx(x, y) ** 2 + exact_dy(x, y) ** 2


def h1_norm(basis, solution, solution_grad):
    """H1 norm of the FEM solution."""
    integration_points = basis.integration_points
    x, y = torch.split(integration_points, 1, dim=-1)

    solution_dx, solution_dy = torch.split(solution_grad, 1, dim=-1)

    return (
        (exact(x, y) - solution) ** 2
        + (exact_dx(x, y) - solution_dx) ** 2
        + (exact_dy(x, y) - solution_dy) ** 2
    )


MESH_SIZE = 0.5**4

fracture_2d_data = {
    "vertices": [
        [-1.0, 0.0],
        [1.0, 0.0],
        [-1.0, 1.0],
        [1.0, 1.0],
        [0.0, 0.0],
        [0.0, 0.5],
        [0.0, 1.0],
    ],
    "segments": [[0, 1], [1, 3], [2, 3], [0, 2], [4, 5], [5, 6]],
}

fracture_triangulation = td.TensorDict(
    tr.triangulate(fracture_2d_data, "pqsea" + str(MESH_SIZE))
)

mesh = MeshTri(triangulation=fracture_triangulation)

elements = ElementTri(polynomial_order=1, integration_order=2)

V = Basis(mesh, elements)

A = V.integrate_bilinear_form(a)

b = V.integrate_linear_form(l)

A_reduced = V.reduce(A)

b_reduced = V.reduce(b)

u_h = torch.zeros(V.basis_parameters["linear_form_shape"])

u_h[V.basis_parameters["inner_dofs"]] = torch.linalg.solve(A_reduced, b_reduced)

exact_value = exact(*torch.unbind(mesh["vertices", "coordinates"], -1))

I_u_h, I_u_h_grad = V.interpolate(V, u_h)

exact_H1_norm = torch.sqrt(torch.sum(V.integrate_functional(h1_exact)))

H1_norm_value = torch.sqrt(
    torch.sum(V.integrate_functional(h1_norm, I_u_h, I_u_h_grad))
)

print((H1_norm_value / exact_H1_norm).item())

fig = plt.figure(figsize=(10, 4), dpi=100)
fig.suptitle(r"FEM computed for $\omega_1$", fontsize=16)

ax1 = fig.add_subplot(1, 3, 1, projection="3d")

ax1.plot_trisurf(
    mesh["vertices", "coordinates"][:, 0],
    mesh["vertices", "coordinates"][:, 1],
    u_h.squeeze(-1),
    triangles=mesh["cells", "vertices"],
    cmap="viridis",
    edgecolor="black",
    linewidth=0.3,
)

ax1.set_title("FEM solution")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel(r"$u_h(x,y)$")

ax2 = fig.add_subplot(1, 3, 2, projection="3d")

ax2.plot_trisurf(
    mesh["vertices", "coordinates"][:, 0],
    mesh["vertices", "coordinates"][:, 1],
    exact_value,
    triangles=mesh["cells", "vertices"],
    cmap="viridis",
    edgecolor="black",
    linewidth=0.3,
)

ax2.set_title("Exact solution")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_zlabel(r"$u(x,y)$")

ax3 = fig.add_subplot(1, 3, 3, projection="3d")

ax3.plot_trisurf(
    mesh["vertices", "coordinates"][:, 0],
    mesh["vertices", "coordinates"][:, 1],
    abs(exact_value - u_h.squeeze(-1)),
    triangles=mesh["cells", "vertices"],
    cmap="viridis",
    edgecolor="black",
    linewidth=0.3,
)

ax3.set_title("Error")
ax3.set_xlabel("x")
ax3.set_ylabel("y")
ax3.set_zlabel(r"$|u-u_h|$")

plt.show()
