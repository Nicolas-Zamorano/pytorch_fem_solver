"""Test fracture jump implementation by comparison with 2D implementation."""

import matplotlib.pyplot as plt
import tensordict as td
import torch
import triangle as tr

from fem import (
    Basis,
    ElementLine,
    ElementTri,
    FractureBasis,
    FracturesTri,
    InteriorEdgesBasis,
    InteriorEdgesFractureBasis,
    MeshTri,
)

torch.set_default_dtype(torch.float64)

MESH_SIZE = 0.5**4

mesh_data = tr.triangulate(
    {"vertices": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]},
    "Dqea" + str(MESH_SIZE),
)

mesh_data_torch = td.TensorDict(mesh_data)

mesh = MeshTri(triangulation=mesh_data_torch)

elements = ElementTri(P_order=1, int_order=4)

V = Basis(mesh, elements)

fig, ax_mesh = plt.subplots()

tr.plot(ax_mesh, **mesh_data)

plt.show()


def rhs(_):
    """Right-hand side function."""
    return 1.0


def rhs_3d(
    _,
):
    """Right-hand side function in 3D."""
    return 1.0


def a(basis):
    """Bilinear form."""
    return basis.v_grad @ basis.v_grad.mT


def l(basis, right_hand_side):
    """Linear form."""
    return right_hand_side(*basis.integration_points) * basis.v


A = V.reduce(V.integrate_bilinear_form(a))
b = V.reduce(V.integrate_linear_form(l, rhs))

u = torch.zeros(V.basis_parameters["linear_form_shape"])
u[V.basis_parameters["inner_dofs"]] = torch.linalg.solve(A, b)

V_inner_edges = InteriorEdgesBasis(mesh, ElementLine(P_order=1, int_order=2))

I_u, I_u_grad = V.interpolate(V_inner_edges, u)

h_E = V.mesh.edges_parameters["inner_edges_length"].unsqueeze(-2)
n_E = V.mesh.edges_parameters["normal4inner_edges"].unsqueeze(-2)


def jump(_, h_element, n_element, solution_grad):
    """Jump functional."""
    solution_grad_plus, solution_grad_minus = torch.unbind(solution_grad, dim=-4)
    return (
        h_element
        * (
            (solution_grad_plus * n_element).sum(-1, keepdim=True)
            + (solution_grad_minus * -n_element).sum(-1, keepdim=True)
        )
        ** 2
    )


eta_E = V_inner_edges.integrate_functional(jump, h_E, n_E, I_u_grad).squeeze(-1)

fractures_triangulation = [mesh_data_torch]

fractures_data = torch.tensor(
    [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]]
)

mesh_f = FracturesTri(
    triangulations=fractures_triangulation, fractures_3D_data=fractures_data
)

elements_f = ElementTri(P_order=1, int_order=2)

V_f = FractureBasis(mesh_f, elements_f)

A_f = V_f.reduce(V_f.integrate_bilinear_form(a))
b_f = V_f.reduce(V_f.integrate_linear_form(l, rhs_3d))

u_f = torch.zeros(V_f.basis_parameters["linear_form_shape"])
u_f[V_f.basis_parameters["inner_dofs"]] = torch.linalg.solve(A_f, b_f)

V_f_inner_edges = InteriorEdgesFractureBasis(
    mesh_f, ElementLine(P_order=1, int_order=2)
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
