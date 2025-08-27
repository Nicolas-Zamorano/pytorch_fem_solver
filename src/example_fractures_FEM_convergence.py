import torch
import pickle

import matplotlib.pyplot as plt
import tensordict as td
import triangle as tr
import numpy as np

from fracture_fem import Fractures, Element_Fracture, Fracture_Basis

torch.set_default_dtype(torch.float64)
# torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
# torch.cuda.empty_cache()

# ---------------------- FEM Parameters ----------------------#

h = 0.5

n = 3

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

fracture_triangulation = tr.triangulate(fracture_2d_data, "pqsea" + str(h ** (n)))

# ---------------------- Residual Parameters ----------------------#


def rhs(x, y, z):

    x_fracture_1, x_fracture_2 = torch.split(x, 1, dim=0)
    y_fracture_1, y_fracture_2 = torch.split(y, 1, dim=0)
    z_fracture_1, z_fracture_2 = torch.split(z, 1, dim=0)

    rhs_fracture_1 = 6.0 * (y_fracture_1 - y_fracture_1**2) * torch.abs(
        x_fracture_1
    ) - 2.0 * (torch.abs(x_fracture_1) ** 3 - torch.abs(x_fracture_1))
    rhs_fracture_2 = -6.0 * (y_fracture_2 - y_fracture_2**2) * torch.abs(
        z_fracture_2
    ) + 2.0 * (torch.abs(z_fracture_2) ** 3 - torch.abs(z_fracture_2))

    rhs_value = torch.cat([rhs_fracture_1, rhs_fracture_2], dim=0)

    return rhs_value


def l(basis):

    x, y, z = basis.integration_points

    v = basis.v
    rhs_value = rhs(x, y, z)

    return rhs_value * v


def a(basis):

    v_grad = basis.v_grad

    return v_grad @ v_grad.mT


# ---------------------- Error Parameters ----------------------#


def exact(x, y, z):

    x_fracture_1, x_fracture_2 = torch.split(x, 1, dim=0)
    y_fracture_1, y_fracture_2 = torch.split(y, 1, dim=0)
    z_fracture_1, z_fracture_2 = torch.split(z, 1, dim=0)

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

    x_fracture_1, x_fracture_2 = torch.split(x, 1, dim=0)
    y_fracture_1, y_fracture_2 = torch.split(y, 1, dim=0)
    z_fracture_1, z_fracture_2 = torch.split(z, 1, dim=0)

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


def H1_exact(basis):

    exact_value = exact(*basis.integration_points)

    exact_dx_value, exact_dy_value, exact_dz_value = torch.split(
        exact_grad(*basis.integration_points), 1, dim=-1
    )

    # return exact_value**2

    return exact_value**2 + exact_dx_value**2 + exact_dy_value**2 + exact_dz_value**2


def H1_norm(basis, I_u_h, I_u_h_grad):

    exact_value = exact(*basis.integration_points)

    exact_dx_value, exact_dy_value, exact_dz_value = torch.split(
        exact_grad(*basis.integration_points), 1, dim=-1
    )

    Ih_x_dx, Ih_x_dy, Ih_x_dz = torch.split(I_u_h_grad, 1, dim=-1)

    L2_error = (exact_value - I_u_h) ** 2

    # return L2_error

    H1_0_error = (
        (exact_dx_value - Ih_x_dx) ** 2
        + (exact_dy_value - Ih_x_dy) ** 2
        + (exact_dz_value - Ih_x_dz) ** 2
    )

    return L2_error + H1_0_error


# ---------------------- Solution ----------------------#

H1_norm_list = []
nb_dofs_list = []

for i in range(11):

    fracture_triangulation = tr.triangulate(
        fracture_triangulation, "prsea" + str(h ** (n + i))
    )

    fracture_triangulation_torch = td.TensorDict((fracture_triangulation))

    fractures_triangulation = (
        fracture_triangulation_torch,
        fracture_triangulation_torch,
    )

    mesh = Fractures(
        triangulations=fractures_triangulation, fractures_3D_data=fractures_data
    )

    elements = Element_Fracture(P_order=1, int_order=2)

    V = Fracture_Basis(mesh, elements)

    exact_H1_norm = torch.sqrt(torch.sum(V.integrate_functional(H1_exact)))

    A = V.integrate_bilinear_form(a)

    b = V.integrate_linear_form(l)

    A_reduced = V.reduce(A)

    b_reduced = V.reduce(b)

    u_h = torch.zeros(V.basis_parameters["linear_form_shape"])

    u_h[V.basis_parameters["inner_dofs"]] = torch.linalg.solve(A_reduced, b_reduced)

    I_u_h, I_u_h_grad = V.interpolate(V, u_h)

    H1_norm_value = (
        torch.sqrt(torch.sum(V.integrate_functional(H1_norm, I_u_h, I_u_h_grad)))
        / exact_H1_norm
    )

    H1_norm_list.append((H1_norm_value).item())

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

with open("H1_norm_converge_FEM.pkl", "wb") as file:
    pickle.dump([nb_dofs_np, H1_norm_np], file)
