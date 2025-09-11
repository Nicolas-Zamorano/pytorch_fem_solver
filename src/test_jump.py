"""Test jump functional implementation against skfem."""

import skfem
import torch
import triangle as tr
from skfem.helpers import dot, grad
import meshio

from fem import Basis, ElementLine, ElementTri, InteriorEdgesBasis, MeshTri

torch.set_default_dtype(torch.float64)

mesh_data = tr.triangulate(
    {"vertices": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]}, "qena0.25"
)

mesh_data_meshio = meshio.Mesh(
    points=mesh_data["vertices"], cells=[("triangle", mesh_data["triangles"])]
)

mesh_sk = skfem.MeshTri(
    doflocs=mesh_data_meshio.points.T,
    t=mesh_data_meshio.cells_dict["triangle"].T,
)

elem_sk = skfem.ElementTriP1()

V_sk = skfem.Basis(mesh_sk, elem_sk, intorder=2)


def f(_):
    """Right-hand side function."""
    return 1.0


@skfem.BilinearForm
def a_sk(u, v, _):
    """Bilinear form."""
    return dot(grad(u), grad(v))


@skfem.LinearForm
def l_sk(v, w):
    """Linear form."""
    return f(w.x) * v


A_sk = a_sk.assemble(V_sk)
b_sk = l_sk.assemble(V_sk)

u_sk = skfem.solve(*skfem.condense(A_sk, b_sk, I=mesh_sk.interior_nodes()))

i_range = [0, 1]

fbasis = [
    skfem.InteriorFacetBasis(mesh_sk, elem_sk, side=i, use_torch=True) for i in i_range
]
u_facet = {"u" + str(i + 1): fbasis[i].interpolate(u_sk) for i in i_range}

I_u_sk = torch.stack([torch.tensor(u_facet["u1"]), torch.tensor(u_facet["u2"])], dim=-1)
I_u_grad_sk = torch.stack(
    [
        torch.tensor(grad(u_facet["u1"])).permute(1, 2, 0),
        torch.tensor(grad(u_facet["u2"])).permute(1, 2, 0),
    ],
    dim=-1,
)


@skfem.Functional
def jump_sk(w):
    """Jump functional."""
    h = w.h
    n = w.n
    dw1 = grad(w["u1"])
    dw2 = grad(w["u2"])
    return h * ((dw1[0] - dw2[0]) * n[0] + (dw1[1] - dw2[1]) * n[1]) ** 2


eta_E_sk = torch.tensor(jump_sk.elemental(fbasis[0], **u_facet))

mesh = MeshTri(mesh_data)

element = ElementTri(polynomial_order=1, integration_order=2)

V = Basis(mesh, element)


def a(basis):
    """Bilinear form."""
    return basis.v_grad @ basis.v_grad.mT


def l(basis):
    """Linear form."""
    return f(basis.integration_points) * basis.v


A = V.reduce(V.integrate_bilinear_form(a))
b = V.reduce(V.integrate_linear_form(l))

sol = torch.zeros(V.basis_parameters["linear_form_shape"])
sol[V.basis_parameters["inner_dofs"]] = torch.linalg.solve(A, b)

element_inner_edges = ElementLine(polynomial_order=1, integration_order=2)

V_inner_edges = InteriorEdgesBasis(mesh, element_inner_edges)

I_u, I_u_grad = V.interpolate(V_inner_edges, sol)

h_E = V.mesh["interior_edges"]["length"].unsqueeze(-2)
n_E = V.mesh["interior_edges"]["normals"].unsqueeze(-2)

I_u_values, I_u_count = torch.unique(
    torch.round(I_u.reshape(-1), decimals=16), return_counts=True
)
I_u_sk_values, I_u_sk_count = torch.unique(
    torch.round(I_u_sk.reshape(-1), decimals=16), return_counts=True
)

I_u_grad_values, I_u_grad_count = torch.unique(
    torch.round(I_u_grad.reshape(-1), decimals=16), return_counts=True
)
I_u_grad_sk_values, I_u_grad_sk_count = torch.unique(
    torch.round(I_u_grad_sk.reshape(-1), decimals=16), return_counts=True
)

print("Values of         I_u:", I_u_values.numpy(), "Repeat:", I_u_count.numpy())
print("Values of      I_u_sk:", I_u_sk_values.numpy(), "Repeat:", I_u_sk_count.numpy())
print(
    "Values of    I_u_grad:", I_u_grad_values.numpy(), "Repeat:", I_u_grad_count.numpy()
)
print(
    "Values of I_u_grad_sk:",
    I_u_grad_sk_values.numpy(),
    "Repeat:",
    I_u_grad_sk_count.numpy(),
)
print(
    "I_u error        norm:",
    (torch.norm(I_u.squeeze(-1).squeeze(-1).mT - I_u_sk) / torch.norm(I_u_sk)).item(),
)
print(
    "I_u_grad         norm:",
    (
        torch.norm(I_u_grad.squeeze(-2) - I_u_grad_sk.mT) / torch.norm(I_u_grad_sk)
    ).item(),
)

print("Size of    I_u_grad: ", I_u_grad.size())
print("Size of I_u_grad_sk: ", I_u_grad_sk.size())


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

print(
    "eta_E            norm:",
    (torch.norm(eta_E.sum() - eta_E_sk.sum()) / torch.norm(eta_E_sk.sum())).item(),
)
