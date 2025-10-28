"""1 fracture 2D test case to compare with fracture implementation."""

import torch

import matplotlib.pyplot as plt
import tensordict as td
import triangle as tr
from mpl_toolkits.mplot3d.art3d import PolyCollection
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


def h1_exact(basis: Basis) -> torch.Tensor:
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

c4e = V.coords4elements

exact_value = exact(*torch.unbind(c4e, -1))

I_u_h, I_u_h_grad = V.interpolate(V, u_h)

exact_H1_norm = torch.sqrt(torch.sum(V.integrate_functional(h1_exact)))

H1_norm_value = torch.sqrt(
    torch.sum(V.integrate_functional(h1_norm, I_u_h, I_u_h_grad))
)

print((H1_norm_value / exact_H1_norm).item())

# Create figure with 3D plot for polynomial order 3 DOFs
fig = plt.figure(figsize=(12, 5), dpi=100)

# 2D plot
ax1 = fig.add_subplot(1, 2, 1)

triangles_plot = PolyCollection(
    c4e,  # type: ignore
    array=u_h.squeeze(-1),
    cmap="viridis",
    edgecolors="black",
    linewidths=0.2,
)

ax1.add_collection(triangles_plot)
ax1.set_title("FEM solution (2D)")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
fig.colorbar(triangles_plot, ax=ax1, label=r"$u_h(x,y)$")

# 3D surface plot using triangulation at DOF nodes
ax2 = fig.add_subplot(1, 2, 2, projection="3d")

# Get coordinates of all DOFs
dof_coords = V.coords4global_dofs  # Shape: (num_dofs, 2)
x_dofs = dof_coords[:, 0].numpy()
y_dofs = dof_coords[:, 1].numpy()
u_dofs = u_h.squeeze(-1).numpy()

# Get the DOF connectivity for each element
dof_connectivity = (
    V.global_dofs4elements.numpy()
)  # Shape: (num_cells, num_dofs_per_cell)

# Create triangulated surface plot
ax2.plot_trisurf(
    x_dofs,
    y_dofs,
    u_dofs,
    triangles=dof_connectivity[:, :3],  # Use first 3 DOFs (vertices) for triangulation
    cmap="viridis",
    edgecolor="black",
    linewidth=0.3,
)

ax2.set_title("FEM solution (3D surface)")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_zlabel(r"$u_h(x,y)$")

plt.show()
