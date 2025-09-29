"""Test file for one fracture in 2D."""

import torch

import matplotlib.pyplot as plt
import tensordict as td
import triangle as tr

from torch_fem import FracturesTri, ElementTri, FractureBasis

# torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
# torch.cuda.empty_cache()
torch.set_default_dtype(torch.float64)

# pylint: disable=not-callable

# ---------------------- FEM Parameters ----------------------#

MESH_SIZE = 0.5 ** (9)

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
    tr.triangulate(fracture_2d_data, "pqsena" + str(MESH_SIZE))
)

fractures_triangulation = [fracture_triangulation]

fractures_data = torch.tensor(
    [
        [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        # [[ 0., 0.,-1.],
        #  [ 0., 0., 1.],
        #  [ 0., 1.,-1.],
        #  [ 0., 1., 1.]]
    ]
)

mesh = FracturesTri(
    triangulations=fractures_triangulation, fractures_3d_data=fractures_data
)

elements = ElementTri(polynomial_order=1, integration_order=2)

V = FractureBasis(mesh, elements)

# ---------------------- Residual Parameters ----------------------#


def rhs(x, y, _):
    """Right-hand side function."""
    x_fracture_1 = x
    y_fracture_1 = y

    rhs_fracture_1 = 6 * x_fracture_1 * (
        y_fracture_1 - y_fracture_1**2
    ) + 2 * x_fracture_1 * (1 - x_fracture_1**2)

    rhs_value = rhs_fracture_1

    return rhs_value


def l(basis):
    """Linear form."""
    x, y, z = torch.split(basis.integration_points, 1, dim=-1)

    v = basis.v
    rhs_value = rhs(x, y, z)

    return rhs_value * v


def a(basis):
    """Bilinear form."""
    v_grad = basis.v_grad

    return v_grad @ v_grad.mT


# ---------------------- Error Parameters ----------------------#


def exact(x, y, _):
    """Exact solution."""
    x_fracture_1 = x
    y_fracture_1 = y

    exact_fracture_1 = (
        y_fracture_1 * (1 - y_fracture_1) * x_fracture_1 * (1 - x_fracture_1**2)
    )

    exact_value = exact_fracture_1

    return exact_value


def exact_dx(x, y, _):
    """Exact solution derivative wrt x."""
    x_fracture_1 = x
    y_fracture_1 = y

    exact_dx_fracture_1 = (
        -y_fracture_1
        * (1 - y_fracture_1)
        * ((x_fracture_1**2 - 1) + 2 * x_fracture_1 * x_fracture_1)
    )

    exact_dx_value = exact_dx_fracture_1

    return exact_dx_value


def exact_dy(x, y, _):
    """Exact solution derivative wrt y."""
    x_fracture_1 = x
    y_fracture_1 = y

    exact_dy_fracture_1 = -(1 - 2 * y_fracture_1) * x_fracture_1 * (x_fracture_1**2 - 1)

    exact_dy_value = exact_dy_fracture_1

    return exact_dy_value


def h1_exact(basis):
    """H1 norm of the exact solution."""
    x, y, z = torch.split(basis.integration_points, 1, dim=-1)
    return exact(x, y, z) ** 2 + exact_dx(x, y, z) ** 2 + exact_dy(x, y, z) ** 2


def h1_norm(basis, solution, solution_grad):
    """H1 norm of the error."""
    solution_dx, solution_dy, _ = torch.split(solution_grad, 1, dim=-1)
    x, y, z = torch.split(basis.integration_points, 1, dim=-1)
    return (
        (exact(x, y, z) - solution) ** 2
        + (exact_dx(x, y, z) - solution_dx) ** 2
        + (exact_dy(x, y, z) - solution_dy) ** 2
    )


# ---------------------- Solution ----------------------#

A = V.integrate_bilinear_form(a)

b = V.integrate_linear_form(l)

u_h = V.solution_tensor()

u_h = V.solve(A, u_h, b)

I_u_h, I_u_h_grad = V.interpolate(V, u_h)

exact_H1_norm = torch.sqrt(torch.sum(V.integrate_functional(h1_exact)))

H1_norm_value = torch.sqrt(
    torch.sum(V.integrate_functional(h1_norm, I_u_h, I_u_h_grad))
)

print((H1_norm_value / exact_H1_norm).item())

# ---------------------- Computation Parameters ----------------------#

NB_FRACTURES = 1

local_vertices_3D = V.global_triangulation["vertices_3D"][
    V.global_triangulation["global2local_idx"].reshape(NB_FRACTURES, -1)
]

exact_value_local = exact(*torch.split(local_vertices_3D, 1, -1)).reshape(-1)

exact_value_global = exact_value_local.reshape(-1, 1)[
    V.global_triangulation["local2global_idx"]
].numpy()

u_h_local = u_h[V.global_triangulation["global2local_idx"]].reshape(-1)
vertices = mesh["vertices", "coordinates"].squeeze(0)
triangles = mesh["cells", "vertices"]

# ---------------------- Plot ----------------------#

fig = plt.figure(figsize=(10, 4), dpi=100)

fig.suptitle(r"FEM computed for $F_1$", fontsize=16)

ax1 = fig.add_subplot(1, 3, 1, projection="3d")

ax1.plot_trisurf(
    vertices[:, 0],
    vertices[:, 1],
    u_h_local,
    triangles=triangles,
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
    vertices[:, 0],
    vertices[:, 1],
    exact_value_local,
    triangles=triangles,
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
    vertices[:, 0],
    vertices[:, 1],
    abs(exact_value_local - u_h_local),
    triangles=triangles,
    cmap="viridis",
    edgecolor="black",
    linewidth=0.3,
)

ax3.set_title("Error")
ax3.set_xlabel("x")
ax3.set_ylabel("y")
ax3.set_zlabel(r"$|u-u_h|$")

plt.show()
