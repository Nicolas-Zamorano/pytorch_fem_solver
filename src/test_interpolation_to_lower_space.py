import meshio
import torch

import matplotlib.pyplot as plt
import triangle as tr
import skfem as fem

from fem import MeshTri, ElementTri, Basis
from skfem.helpers import grad, dot

vertices = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]

h = 0.5 ** (1)

mesh_H = tr.triangulate(dict(vertices=vertices), "Dqeca" + str(h))

##------------------ fem ------------------##

V_H = Basis(MeshTri(triangulation=mesh_H), ElementTri(P_order=2, int_order=3))

f = lambda x, y: 1.0


def a(elements):
    return elements.v_grad @ elements.v_grad.mT


def l(elements):
    return f(*elements.integration_points) * elements.v


A = V_H.reduce(V_H.integrate_bilinear_form(a))
b = V_H.reduce(V_H.integrate_linear_form(l))

u = torch.zeros(V_H.basis_parameters["linear_form_shape"])
u[V_H.basis_parameters["inner_dofs"]] = torch.linalg.solve(A, b)

mesh_h = tr.triangulate(
    dict(vertices=V_H.coords4global_dofs, segments=mesh_H["edges"]), "Dqepca" + str(h)
)

V_h = Basis(MeshTri(triangulation=mesh_h), ElementTri(P_order=1, int_order=3))

I_H_u, I_H_u_grad = V_H.interpolate(V_h, u)


tr.compare(plt, mesh_H, mesh_h)

plt.show()

##------------------ scikit-fem ------------------##

# mesh_sk_H = fem.MeshTri1().refined(0)

meshio.Mesh(points=mesh_H["vertices"], cells=[("triangle", mesh_H["triangles"])]).write(
    "mesh_H.msh"
)

mesh_sk_H = fem.MeshTri().load("mesh_H.msh")

V_H_sk = fem.Basis(mesh_sk_H, fem.ElementTriP2(), intorder=3)


@fem.BilinearForm
def a_sk(u, v, w):
    return dot(grad(u), grad(v))


@fem.LinearForm
def l_sk(v, w):
    return f(*w.x) * v


A_sk = a_sk.assemble(V_H_sk)
b_sk = l_sk.assemble(V_H_sk)

u_sk = fem.solve(*fem.condense(A_sk, b_sk, D=V_H_sk.get_dofs()))
u_sk_torch = torch.tensor(u_sk).unsqueeze(-1)

# mesh_sk_h = mesh_sk_H.refined(1)

meshio.Mesh(points=mesh_h["vertices"], cells=[("triangle", mesh_h["triangles"])]).write(
    "mesh_h.msh"
)

mesh_sk_h = fem.MeshTri().load("mesh_h.msh")

V_h_sk = fem.Basis(mesh_sk_h, fem.ElementTriP1(), intorder=3)

I_H_u_sk = torch.tensor(V_h_sk.interpolate(u_sk))
I_H_u_grad_sk = torch.tensor(grad(V_h_sk.interpolate(u_sk)))

##------------------ values ------------------##

I_u_values, I_u_count = torch.unique(I_H_u.reshape(-1), return_counts=True)
I_u_sk_values, I_u_sk_count = torch.unique(I_H_u_sk.reshape(-1), return_counts=True)

print("Values of I_u   ", I_u_values.numpy(), "Repeat:", I_u_count.numpy())
print("Values of I_u_sk", I_u_sk_values.numpy(), "Repeat:", I_u_sk_count.numpy())

I_u_grad_values, I_u_grad_count = torch.unique(
    I_H_u_grad.reshape(-1), return_counts=True
)
I_u_grad_sk_values, I_u_grad_sk_count = torch.unique(
    I_H_u_grad_sk.reshape(-1), return_counts=True
)

print(
    "Values of I_u_grad   ", I_u_grad_values.numpy(), "Repeat:", I_u_grad_count.numpy()
)
print(
    "Values of I_u_grad_sk",
    I_u_grad_sk_values.numpy(),
    "Repeat:",
    I_u_grad_sk_count.numpy(),
)

print("u error   norm:", (torch.norm(u_sk_torch - u) / torch.norm(u_sk_torch)))
print(
    "I_u error norm:",
    (
        torch.norm(I_H_u.squeeze(-1).squeeze(-1) - I_H_u_sk) / torch.norm(I_H_u_sk)
    ).item(),
)
print(
    "I_u_grad  norm:",
    (
        torch.norm(I_H_u_grad.squeeze(-2).permute((2, 0, 1)) - I_H_u_grad_sk)
        / torch.norm(I_H_u_grad_sk)
    ).item(),
)
