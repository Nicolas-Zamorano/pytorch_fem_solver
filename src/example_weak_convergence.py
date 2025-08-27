import torch

import matplotlib.pyplot as plt
import tensordict as td
import triangle as tr
import numpy as np

from fem import MeshTri, ElementTri, Basis

h = 0.5

n = 3

pts = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]

segments = tr.convex_hull(pts)

fracture_triangulation = tr.triangulate(
    dict(vertices=pts, segments=segments), "pqsea" + str(h ** (n))
)


# ---------------------- Residual Parameters ----------------------#

rhs = lambda x, y: 6 * x * (y - y**2) + 2 * x * (1 - x**2)


def l(basis):

    x, y = basis.integration_points

    v = basis.v
    rhs_value = rhs(x, y)

    return rhs_value * v


def a(basis):

    v_grad = basis.v_grad

    return v_grad @ v_grad.mT


# ---------------------- Error Parameters ----------------------#

exact = lambda x, y: y * (1 - y) * x * (1 - x**2)
exact_dx = (
    lambda x, y: -y * (1 - y) * (torch.sign(x) * (x**2 - 1) + 2 * x * torch.abs(x))
)
exact_dy = lambda x, y: -(1 - 2 * y) * torch.abs(x) * (x**2 - 1)


def H1_exact(basis):

    x, y = basis.integration_points

    return exact(x, y) ** 2 + exact_dx(x, y) ** 2 + exact_dy(x, y) ** 2


def H1_norm(basis, Ih_u, Ih_u_grad):

    x, y = basis.integration_points

    Ih_u_dx, Ih_u_dy = torch.split(Ih_u_grad, 1, dim=-1)

    return (
        (exact(x, y) - Ih_u) ** 2
        + (exact_dx(x, y) - Ih_u_dx) ** 2
        + (exact_dy(x, y) - Ih_u_dy) ** 2
    )


H1_error_list = []

# ---------------------- Solution ----------------------#

H1_norm_list = []
nb_dofs_list = []

for i in range(11):

    fracture_triangulation = tr.triangulate(
        fracture_triangulation, "pqrsea" + str(h ** (n + i))
    )

    fracture_triangulation_torch = td.TensorDict((fracture_triangulation))

    mesh = MeshTri(fracture_triangulation_torch)

    elements = ElementTri(P_order=1, int_order=4)

    V = Basis(mesh, elements)

    exact_H1_norm = torch.sqrt(torch.sum(V.integrate_functional(H1_exact)))

    A = V.integrate_bilinear_form(a)

    b = V.integrate_linear_form(l)

    A_reduced = V.reduce(A)

    b_reduced = V.reduce(b)

    u_h = torch.zeros(V.basis_parameters["linear_form_shape"])

    u_h[V.basis_parameters["inner_dofs"]] = torch.linalg.solve(A_reduced, b_reduced)

    I_u_h, I_u_h_grad = V.interpolate(V, u_h)

    H1_norm_value = torch.sqrt(
        torch.sum(V.integrate_functional(H1_norm, I_u_h, I_u_h_grad))
    )

    H1_norm_list.append((H1_norm_value / exact_H1_norm).item())

    nb_dofs_list.append(V.basis_parameters["nb_dofs"])

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
