"""Example of FEM convergence on fractures."""

import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import triangle as tr
from torch_fem import FracturesTri, ElementTri, FractureBasis

torch.set_default_dtype(torch.float64)
# torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
# torch.cuda.empty_cache()

# ---------------------- FEM Parameters ----------------------#

ELEMENT_SIZE = 0.5

EXPONENT = 3

fracture_2d_data = {
    "vertices": [
        [-1.0, 0.0],
        [1.0, 0.0],
        [-1.0, 1.0],
        [1.0, 1.0],
        [0.0, 0.0],
        [0.0, 1.0],
    ],
    "segments": [[0, 1], [1, 3], [2, 3], [0, 2], [4, 5]],
}

fractures_data = torch.tensor(
    [
        [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        [[0.0, 0.0, -1.0], [0.0, 0.0, 1.0], [0.0, 1.0, -1.0], [0.0, 1.0, 1.0]],
    ]
)

fracture_triangulation = tr.triangulate(
    fracture_2d_data, "pqsea" + str(ELEMENT_SIZE ** (EXPONENT))
)

# ---------------------- Residual Parameters ----------------------#


def rhs(x, y, z):
    """Right-hand side function."""
    x_fracture_1, _ = torch.split(x, 1, dim=0)
    y_fracture_1, y_fracture_2 = torch.split(y, 1, dim=0)
    _, z_fracture_2 = torch.split(z, 1, dim=0)

    rhs_fracture_1 = 6.0 * (y_fracture_1 - y_fracture_1**2) * torch.abs(
        x_fracture_1
    ) - 2.0 * (torch.abs(x_fracture_1) ** 3 - torch.abs(x_fracture_1))
    rhs_fracture_2 = -6.0 * (y_fracture_2 - y_fracture_2**2) * torch.abs(
        z_fracture_2
    ) + 2.0 * (torch.abs(z_fracture_2) ** 3 - torch.abs(z_fracture_2))

    rhs_value = torch.cat([rhs_fracture_1, rhs_fracture_2], dim=0)

    return rhs_value


def l(basis):
    """Linear form."""
    x, y, z = basis.integration_points

    v = basis.v
    rhs_value = rhs(x, y, z)

    return rhs_value * v


def a(basis):
    """Bilinear form."""
    v_grad = basis.v_grad

    return v_grad @ v_grad.mT


# ---------------------- Error Parameters ----------------------#


def exact(x, y, z):
    """Exact solution."""
    x_fracture_1, _ = torch.split(x, 1, dim=0)
    y_fracture_1, y_fracture_2 = torch.split(y, 1, dim=0)
    _, z_fracture_2 = torch.split(z, 1, dim=0)

    exact_fracture_1 = (
        -y_fracture_1
        * (1 - y_fracture_1)
        * torch.abs(x_fracture_1)
        * (x_fracture_1**2 - 1)
    )
    exact_fracture_2 = (
        y_fracture_2
        * (1 - y_fracture_2)
        * torch.abs(z_fracture_2)
        * (z_fracture_2**2 - 1)
    )

    exact_value = torch.cat([exact_fracture_1, exact_fracture_2], dim=0)

    return exact_value


def exact_grad(x, y, z):
    """Gradient of the exact solution."""
    x_fracture_1, _ = torch.split(x, 1, dim=0)
    y_fracture_1, y_fracture_2 = torch.split(y, 1, dim=0)
    _, z_fracture_2 = torch.split(z, 1, dim=0)

    exact_dx_fracture_1 = (
        -y_fracture_1
        * (1 - y_fracture_1)
        * (
            torch.sign(x_fracture_1) * (x_fracture_1**2 - 1)
            + 2 * x_fracture_1 * torch.abs(x_fracture_1)
        )
    )
    exact_dy_fracture_1 = (
        -(1 - 2 * y_fracture_1) * torch.abs(x_fracture_1) * (x_fracture_1**2 - 1)
    )
    exact_dz_fracture_1 = torch.zeros_like(exact_dx_fracture_1)

    exact_grad_fracture_1 = torch.cat(
        [exact_dx_fracture_1, exact_dy_fracture_1, exact_dz_fracture_1], dim=-1
    )

    exact_dy_fracture_2 = (
        (1 - 2 * y_fracture_2) * torch.abs(z_fracture_2) * (z_fracture_2**2 - 1)
    )
    exact_dz_fracture_2 = (
        y_fracture_2
        * (1 - y_fracture_2)
        * (
            torch.sign(z_fracture_2) * (z_fracture_2**2 - 1)
            + 2 * z_fracture_2 * torch.abs(z_fracture_2)
        )
    )
    exact_dx_fracture_2 = torch.zeros_like(exact_dz_fracture_2)

    exact_grad_fracture_2 = torch.cat(
        [exact_dx_fracture_2, exact_dy_fracture_2, exact_dz_fracture_2], dim=-1
    )

    grad_value = torch.cat([exact_grad_fracture_1, exact_grad_fracture_2], dim=0)

    return grad_value


def h1_exact(basis):
    """H1 norm of the exact solution."""
    exact_value = exact(*basis.integration_points)

    exact_dx_value, exact_dy_value, exact_dz_value = torch.split(
        exact_grad(*basis.integration_points), 1, dim=-1
    )

    # return exact_value**2

    return exact_value**2 + exact_dx_value**2 + exact_dy_value**2 + exact_dz_value**2


def h1_norm(basis, solution, solution_grad):
    """H1 norm of the error."""
    exact_value = exact(*basis.integration_points)

    exact_dx_value, exact_dy_value, exact_dz_value = torch.split(
        exact_grad(*basis.integration_points), 1, dim=-1
    )

    solution_dx, solution_dy, solution_dz = torch.split(solution_grad, 1, dim=-1)

    l2_error = (exact_value - solution) ** 2

    # return L2_error

    h1_0_error = (
        (exact_dx_value - solution_dx) ** 2
        + (exact_dy_value - solution_dy) ** 2
        + (exact_dz_value - solution_dz) ** 2
    )

    return l2_error + h1_0_error


# ---------------------- Solution ----------------------#

H1_norm_list = []
nb_dofs_list = []

for i in range(11):

    fracture_triangulation = tr.triangulate(
        fracture_triangulation, "prsea" + str(ELEMENT_SIZE ** (EXPONENT + i))
    )

    fractures_triangulation = [
        fracture_triangulation,
        fracture_triangulation,
    ]

    mesh = FracturesTri(
        triangulations=fractures_triangulation, fractures_3d_data=fractures_data
    )

    elements = ElementTri(polynomial_order=1, integration_order=2)

    V = FractureBasis(mesh, elements)

    exact_H1_norm = torch.sqrt(torch.sum(V.integrate_functional(h1_exact)))

    A = V.integrate_bilinear_form(a)

    b = V.integrate_linear_form(l)

    u_h = V.solution_tensor()

    u_h = V.solve(A, u_h, b)

    I_u_h, I_u_h_grad = V.interpolate(V, u_h)

    H1_norm_value = (
        torch.sqrt(torch.sum(V.integrate_functional(h1_norm, I_u_h, I_u_h_grad)))
        / exact_H1_norm
    )

    H1_norm_list.append((H1_norm_value).item())

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

with open("H1_norm_converge_FEM.pkl", "wb") as file:
    pickle.dump([nb_dofs_np, H1_norm_np], file)
