"""1 fracture 2D test case to compare with fracture implementation."""

import torch

import matplotlib.pyplot as plt
import tensordict as td
import triangle as tr
from mpl_toolkits.mplot3d.art3d import PolyCollection
from torch_fem import MeshTri, ElementTri, Basis

torch.set_default_dtype(torch.float64)

# pylint: disable=not-callable


def rhs(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Right-hand side function."""
    return 6 * x * (y - y**2) + 2 * x * (1 - x**2)


def l(basis: Basis) -> torch.Tensor:
    """Linear form."""
    integration_points = basis.integration_points
    x, y = torch.split(integration_points, 1, dim=-1)

    v = basis.v

    return rhs(x, y) * v


def a(basis: Basis) -> torch.Tensor:
    """Bilinear form."""
    v_grad = basis.v_grad

    return v_grad @ v_grad.mT


def exact(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Exact solution."""
    return y * (1 - y) * x * (1 - x**2)


def exact_dx(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Derivative of the exact solution with respect to x."""
    return -y * (1 - y) * ((x**2 - 1) + 2 * x * x)


def exact_dy(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Derivative of the exact solution with respect to y."""
    return -(1 - 2 * y) * x * (x**2 - 1)


def h1_exact(basis: Basis) -> torch.Tensor:
    """H1 norm of the exact solution."""
    integration_points = basis.integration_points
    x, y = torch.split(integration_points, 1, dim=-1)

    return exact(x, y) ** 2 + exact_dx(x, y) ** 2 + exact_dy(x, y) ** 2


def h1_norm(
    basis: Basis, solution: torch.Tensor, solution_grad: torch.Tensor
) -> torch.Tensor:
    """H1 norm of the FEM solution."""
    integration_points = basis.integration_points
    x, y = torch.split(integration_points, 1, dim=-1)

    solution_dx, solution_dy = torch.split(solution_grad, 1, dim=-1)

    return (
        (exact(x, y) - solution) ** 2
        + (exact_dx(x, y) - solution_dx) ** 2
        + (exact_dy(x, y) - solution_dy) ** 2
    )


MESH_SIZE = 0.5**10

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

elements = ElementTri(polynomial_order=3, integration_order=6)

V = Basis(mesh, elements)

A = V.integrate_bilinear_form(a)

b = V.integrate_linear_form(l)

u_h = V.solve(A, b)

c4e = V.coords_4_elements

exact_value = exact(*torch.unbind(c4e, -1))

I_u_h, I_u_h_grad = V.interpolate(V, u_h)

exact_H1_norm = torch.sqrt(torch.sum(V.integrate_functional(h1_exact)))

H1_norm_value = torch.sqrt(
    torch.sum(V.integrate_functional(h1_norm, I_u_h, I_u_h_grad))
)

print((H1_norm_value / exact_H1_norm).item())

### --- PLOT 2D --- ###

fig_2d, ax_2d = plt.subplots()

triangles_plot = PolyCollection(
    c4e,  # type: ignore
    array=I_u_h.mean(dim=-3).reshape(-1),
    cmap="viridis",
    edgecolors="black",
    linewidths=0.2,
)

ax_2d.add_collection(triangles_plot)
ax_2d.set_title("FEM solution (2D)")
ax_2d.set_xlabel("x")
ax_2d.set_ylabel("y")
ax_2d.set_xlim((-1, 1))
ax_2d.set_ylim((0, 1))
fig_2d.colorbar(triangles_plot, ax=ax_2d, label=r"$u_h(x,y)$")

### --- PLOT 3D --- ###

fig_3d, ax_3d = plt.subplots(subplot_kw={"projection": "3d"})

dof_coords = V.coords_4_global_dofs
x_dofs = dof_coords[:, 0]
y_dofs = dof_coords[:, 1]
u_dofs = u_h.squeeze(-1)

# Get the DOF connectivity for each element

# Create triangulated surface plot
ax_3d.plot_trisurf(
    x_dofs,
    y_dofs,
    u_dofs,
    triangles=V.global_dofs_4_elements[:, :3],
    cmap="viridis",
    edgecolor="black",
    linewidth=0.3,
)

ax_3d.set_title("FEM solution (3D surface)")
ax_3d.set_xlabel("x")
ax_3d.set_ylabel("y")
ax_3d.set_zlabel(r"$u_h(x,y)$")

plt.show()
