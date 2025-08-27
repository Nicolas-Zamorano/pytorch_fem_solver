import torch
from fracture_fem import (
    Fractures,
    Element_Fracture,
    Fracture_Basis,
    Fracture_Element_Line,
    Interior_Facet_Fracture_Basis,
)
from fem import MeshTri, ElementTri, Basis, ElementLine, InteriorFacetBasis
import triangle as tr
import tensordict as td
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)

h = 0.5**4

mesh_data = tr.triangulate(
    {"vertices": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]}, "Dqea" + str(h)
)

mesh_data_torch = td.TensorDict(mesh_data)

mesh = MeshTri(triangulation=mesh_data_torch)

elements = ElementTri(P_order=1, int_order=4)

V = Basis(mesh, elements)

fig, ax_mesh = plt.subplots()

tr.plot(ax_mesh, **mesh_data)

plt.show()

f = lambda x, y: 1.0
f_3D = lambda x, y, z: 1


def a(elements):
    return elements.v_grad @ elements.v_grad.mT


def l(elements, rhs):
    return rhs(*elements.integration_points) * elements.v


A = V.reduce(V.integrate_bilinear_form(a))
b = V.reduce(V.integrate_linear_form(l, f))

u = torch.zeros(V.basis_parameters["linear_form_shape"])
u[V.basis_parameters["inner_dofs"]] = torch.linalg.solve(A, b)

V_inner_edges = InteriorFacetBasis(mesh, ElementLine(P_order=1, int_order=2))

I_u, I_u_grad = V.interpolate(V_inner_edges, u)

h_E = V.mesh.edges_parameters["inner_edges_length"].unsqueeze(-2)
n_E = V.mesh.edges_parameters["normal4inner_edges"].unsqueeze(-2)


def jump(elements, h_E, n_E, I_u_grad):
    I_u_grad_plus, I_u_grad_minus = torch.unbind(I_u_grad, dim=-4)
    return (
        h_E
        * (
            (I_u_grad_plus * n_E).sum(-1, keepdim=True)
            + (I_u_grad_minus * -n_E).sum(-1, keepdim=True)
        )
        ** 2
    )


eta_E = V_inner_edges.integrate_functional(jump, h_E, n_E, I_u_grad).squeeze(-1)

fractures_triangulation = [mesh_data_torch]

fractures_data = torch.tensor(
    [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]]
)

mesh_f = Fractures(
    triangulations=fractures_triangulation, fractures_3D_data=fractures_data
)

elements_f = Element_Fracture(P_order=1, int_order=2)

V_f = Fracture_Basis(mesh_f, elements_f)

A_f = V_f.reduce(V_f.integrate_bilinear_form(a))
b_f = V_f.reduce(V_f.integrate_linear_form(l, f_3D))

u_f = torch.zeros(V_f.basis_parameters["linear_form_shape"])
u_f[V_f.basis_parameters["inner_dofs"]] = torch.linalg.solve(A_f, b_f)

V_f_inner_edges = Interior_Facet_Fracture_Basis(
    mesh_f, Fracture_Element_Line(P_order=1, int_order=2)
)

I_u_f, I_u_f_grad = V_f.interpolate(V_f_inner_edges, u_f)

h_E_f = V_f.mesh.edges_parameters["inner_edges_length"].unsqueeze(-2)
n_E_f = V_f.mesh.local_triangulations["normal4inner_edges_3D"]

I_u_values, I_u_count = torch.unique(
    torch.round(I_u.reshape(-1), decimals=12), return_counts=True
)
I_u_f_values, I_u_f_count = torch.unique(
    torch.round(I_u_f.reshape(-1), decimals=12), return_counts=True
)

I_u_grad_values, I_u_grad_count = torch.unique(
    torch.round(I_u_grad.reshape(-1), decimals=12), return_counts=True
)
I_u_f_grad_values, I_u_f_grad_count = torch.unique(
    torch.round(I_u_f_grad[..., :2].reshape(-1), decimals=12), return_counts=True
)

print("Values of        I_u:", I_u_values.numpy(), "Repeat:", I_u_count.numpy())
print("Values of      I_u_f:", I_u_f_values.numpy(), "Repeat:", I_u_f_count.numpy())
print(
    "Values of   I_u_grad:", I_u_grad_values.numpy(), "Repeat:", I_u_grad_count.numpy()
)
print(
    "Values of I_u_f_grad:",
    I_u_f_grad_values.numpy(),
    "Repeat:",
    I_u_f_grad_count.numpy(),
)
print(
    "I_u error       norm:",
    (torch.norm(I_u - I_u_f.squeeze(0)) / torch.norm(I_u)).item(),
)
print(
    "I_u_grad        norm:",
    (
        torch.norm(I_u_grad - I_u_f_grad[..., :2].squeeze(0)) / torch.norm(I_u_grad)
    ).item(),
)

eta_E_f = V_inner_edges.integrate_functional(jump, h_E_f, n_E_f, I_u_f_grad)

print(
    "eta_E           norm:",
    (torch.norm(eta_E - eta_E_f.squeeze(-1)) / torch.norm(eta_E)).item(),
)
