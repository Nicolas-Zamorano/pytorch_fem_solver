"""Test assembly of PatchesBasis"""

import math

import torch
import triangle as tr

from torch_fem import Basis, ElementTri, MeshTri, Patches, PatchesBasis

centers = torch.tensor([[0.5, 0.5]])

radius = torch.tensor([[0.5]])

mesh_data = tr.triangulate(
    {"vertices": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]]}, "qne"
)

patches = Patches(centers, radius)

mesh = MeshTri(mesh_data)

elements = ElementTri(1, 2)

V = Basis(mesh, elements)

V_patches = PatchesBasis(patches, elements)


def bilinear(basis):
    """Stiffness + Mass matrix."""
    v = basis.v
    v_grad = basis.v_grad

    return v_grad @ v_grad.mT + v @ v.mT


def rhs(x, y):
    """Right-hand side function."""
    return 2.0 * math.pi**2 * torch.sin(math.pi * x) * torch.sin(math.pi * y)


def residual(basis):
    """Load vector."""
    x, y = torch.split(basis.integration_points, 1, dim=-1)
    v = basis.v

    return rhs(x, y) * v


def rhs_functional(basis):
    """Integral of the squared rhs."""
    x, y = torch.split(basis.integration_points, 1, dim=-1)

    return rhs(x, y) ** 2


stiff_matrix = V.reduce(V.integrate_bilinear_form(bilinear))

rhs_vector = V.integrate_linear_form(residual)

stiff_matrix_patches = V_patches.reduce(
    V_patches.integrate_bilinear_form(bilinear)
).squeeze(0)

rhs_vector_patches = V_patches.integrate_linear_form(residual).squeeze(0)

zeros = torch.zeros(1)

assert torch.isclose(
    torch.norm(stiff_matrix - stiff_matrix_patches) / torch.norm(stiff_matrix), zeros
)
assert torch.isclose(
    torch.norm(rhs_vector - rhs_vector_patches) / torch.norm(rhs_vector), zeros
)
