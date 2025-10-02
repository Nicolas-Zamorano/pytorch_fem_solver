"""Example of weak convergence for a manufactured solution on the unit square."""

import torch

import matplotlib.pyplot as plt
import tensordict as td
import triangle as tr
import numpy as np

from torch_fem import MeshTri, ElementTri, Basis

MESH_SIZE = 0.5

EXPONENT = 3

pts = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]

segments = tr.convex_hull(pts)

fracture_triangulation = tr.triangulate(
    dict(vertices=pts, segments=segments), "pqsnea" + str(MESH_SIZE ** (EXPONENT))
)


# ---------------------- Residual Parameters ----------------------#


def rhs(x, y):
    """Right-hand side function."""
    return 6 * x * (y - y**2) + 2 * x * (1 - x**2)


def l(basis):
    """Linear form."""
    x, y = basis.integration_points

    v = basis.v
    rhs_value = rhs(x, y)

    return rhs_value * v


def a(basis):
    """Bilinear form."""
    v_grad = basis.v_grad

    return v_grad @ v_grad.mT


# ---------------------- Error Parameters ----------------------#


def exact(x, y):
    """Exact solution."""
    return y * (1 - y) * x * (1 - x**2)


def exact_dx(x, y):
    """Derivative of the exact solution with respect to x."""
    return -y * (1 - y) * (torch.sign(x) * (x**2 - 1) + 2 * x * torch.abs(x))


def exact_dy(x, y):
    """Derivative of the exact solution with respect to y."""
    return -(1 - 2 * y) * torch.abs(x) * (x**2 - 1)


def h1_exact(basis):
    """H1 norm of the exact solution."""
    x, y = basis.integration_points

    return exact(x, y) ** 2 + exact_dx(x, y) ** 2 + exact_dy(x, y) ** 2


def h1_norm(basis, solution, solution_grad):
    """H1 norm of the error."""
    x, y = basis.integration_points

    solution_dx, solution_dy = torch.split(solution_grad, 1, dim=-1)

    return (
        (exact(x, y) - solution) ** 2
        + (exact_dx(x, y) - solution_dx) ** 2
        + (exact_dy(x, y) - solution_dy) ** 2
    )


H1_error_list = []

# ---------------------- Solution ----------------------#

H1_norm_list = []
nb_dofs_list = []

for i in range(11):

    fracture_triangulation = tr.triangulate(
        fracture_triangulation, "pqrsea" + str(MESH_SIZE ** (EXPONENT + i))
    )

    fracture_triangulation_torch = td.TensorDict((fracture_triangulation))

    mesh = MeshTri(fracture_triangulation_torch)

    elements = ElementTri(polynomial_order=1, integration_order=4)

    V = Basis(mesh, elements)

    exact_H1_norm = torch.sqrt(torch.sum(V.integrate_functional(h1_exact)))

    A = V.integrate_bilinear_form(a)

    b = V.integrate_linear_form(l)

    u_h = V.solution_tensor()

    u_h = V.solve(A, u_h, b)

    I_u_h, I_u_h_grad = V.interpolate(V, u_h)

    H1_norm_value = torch.sqrt(
        torch.sum(V.integrate_functional(h1_norm, I_u_h, I_u_h_grad))
    )

    H1_norm_list.append((H1_norm_value / exact_H1_norm).item())

    nb_dofs_list.append(u_h.shape[-2])

nb_dofs_np = np.array(nb_dofs_list)
H1_norm_np = np.array(H1_norm_list)

log_dofs = np.log10(nb_dofs_np)
log_H1 = np.log10(H1_norm_np)
slope, intercept = np.polyfit(log_dofs, log_H1, 1)

H1_fit = 10 ** (intercept) * nb_dofs_np**slope

fig, ax = plt.subplots(dpi=500)
ax.loglog(nb_dofs_np, H1_norm_np, "x")
ax.loglog(nb_dofs_np, H1_fit, "--")

ax.set_xlabel("# DOFs")
ax.set_ylabel("relative error")
plt.show()
