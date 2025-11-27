"""Test interpolation to lower space."""

import matplotlib.pyplot as plt
import meshio
import skfem as fem
import torch
import triangle as tr
from skfem.helpers import dot, grad
import numpy as np

from torch_fem import Basis, ElementTri, MeshTri

vertices = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]

MESH_SIZE = 0.5 ** (1)

mesh_H_data = tr.triangulate({"vertices": vertices}, "qena" + str(MESH_SIZE))

K_INT = 2
K_TEST = 1
Q = 4

##------------------ fem ------------------##

V_H = Basis(
    MeshTri(triangulation=mesh_H_data),
    ElementTri(polynomial_order=K_INT, integration_order=Q),
)


def rhs(x, y=0):
    """Right-hand side function."""
    return 1.0


def a(elements):
    """Bilinear form."""
    return elements.v_grad @ elements.v_grad.mT


def l(elements):
    """Linear form."""
    return rhs(elements.integration_points) * elements.v


A = V_H.integrate_bilinear_form(a)
b = V_H.integrate_linear_form(l)

u_H = V_H.solve(A, b)

regions = [
    [c[0], c[1], int(i), 0]
    for i, c in enumerate(V_H.mesh["cells", "coordinates"].mean(-2))
]

mesh_h = tr.triangulate(
    {
        "vertices": V_H.coordinates_4_global_dofs,
        "segments": V_H.vertices_4_new_edges,
        "regions": regions,
    },
    "penA" + str(MESH_SIZE),
)

V_h = Basis(
    MeshTri(triangulation=mesh_h),
    ElementTri(polynomial_order=K_TEST, integration_order=Q),
)

I_H_u, I_H_u_grad = V_H.interpolate(V_h, u_H)

I_H_u = I_H_u.sum(-3)
I_H_u_grad = I_H_u_grad.sum(-3)

# tr.compare(plt, mesh_H, mesh_h)

# plt.show()

##------------------ scikit-fem ------------------##

meshio.Mesh(
    points=mesh_H_data["vertices"], cells=[("triangle", mesh_H_data["triangles"])]
).write("mesh_H.msh")

mesh_sk_H = fem.MeshTri().load("mesh_H.msh")

V_H_sk = fem.Basis(mesh_sk_H, fem.ElementTriP2(), intorder=Q)


@fem.BilinearForm
def a_sk(u, v, _):
    """Bilinear form."""
    return dot(grad(u), grad(v))


@fem.LinearForm
def l_sk(v, w):
    """Linear form."""
    return rhs(*w.x) * v


A_sk = a_sk.assemble(V_H_sk)
b_sk = l_sk.assemble(V_H_sk)

u_sk = fem.solve(*fem.condense(A_sk, b_sk, D=V_H_sk.get_dofs()))
u_sk_torch = torch.tensor(u_sk).unsqueeze(-1)

# mesh_sk_h = mesh_sk_H.refined(1)

meshio.Mesh(points=mesh_h["vertices"], cells=[("triangle", mesh_h["triangles"])]).write(
    "mesh_h.msh"
)

mesh_sk_h = fem.MeshTri().load("mesh_h.msh")

V_h_sk = fem.Basis(mesh_sk_h, fem.ElementTriP1(), intorder=Q)


@fem.Functional
def inter(w):
    """Interpolate u_sk to V_h_sk."""
    interpolation_func = V_h_sk.interpolator(u_sk)
    x = np.stack(w.x, axis=0)
    return interpolation_func(x)


@fem.Functional
def inter_grad(w):
    """Interpolate grad u_sk to V_h_sk."""
    interpolation_func = V_h_sk.interpolator(u_sk)
    x = np.stack(w.x, axis=0)
    return grad(interpolation_func)(x)


I_H_u_sk = torch.tensor(inter.elemental(V_h_sk))
# I_H_u_grad_sk = torch.tensor(inter_grad.elemental(V_h_sk))


##------------------ values ------------------##

I_u_values, I_u_count = torch.unique(
    torch.round(I_H_u.reshape(-1), decimals=3), return_counts=True
)
I_u_sk_values, I_u_sk_count = torch.unique(
    torch.round(I_H_u_sk.reshape(-1), decimals=3), return_counts=True
)

print(
    "Values of I_u   ",
    I_u_values.numpy(),
    "Repeat:",
    I_u_count.numpy(),
)
print(
    "Values of I_u_sk",
    I_u_sk_values.numpy(),
    "Repeat:",
    I_u_sk_count.numpy(),
)

I_u_grad_values, I_u_grad_count = torch.unique(
    torch.round(I_H_u_grad.reshape(-1), decimals=3), return_counts=True
)
# I_u_grad_sk_values, I_u_grad_sk_count = torch.unique(
#     torch.round(I_H_u_grad_sk.reshape(-1), decimals=3), return_counts=True
# )

print(
    "Values of I_u_grad   ", I_u_grad_values.numpy(), "Repeat:", I_u_grad_count.numpy()
)
# print(
#     "Values of I_u_grad_sk",
#     I_u_grad_sk_values.numpy(),
#     "Repeat:",
#     I_u_grad_sk_count.numpy(),
# )

print("u error   norm:", (torch.norm(u_sk_torch - u_H) / torch.norm(u_sk_torch)))
print(
    "I_u error norm:",
    (
        torch.norm(I_H_u.squeeze(-1).squeeze(-1) - I_H_u_sk) / torch.norm(I_H_u_sk)
    ).item(),
)
# print(
#     "I_u_grad  norm:",
#     (
#         torch.norm(I_H_u_grad.squeeze(-2).permute((2, 0, 1)) - I_H_u_grad_sk)
#         / torch.norm(I_H_u_grad_sk)
#     ).item(),
# )
