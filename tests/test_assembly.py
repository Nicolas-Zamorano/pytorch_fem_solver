"""Tests for assembly of matrices and vectors."""

import math

import meshio
import numpy as np
import skfem
import torch
import triangle as tr
from skfem.helpers import dot, grad

from tensordict import TensorDict

from torch_fem import Basis, ElementTri, MeshTri

# pyright: reportArgumentType=false


def test_assembly():
    """Test the assembly of matrices and vectors against scikit-fem."""
    torch.set_default_dtype(torch.float64)

    mesh_data = tr.triangulate(
        {"vertices": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]}, "qea0.5"
    )

    mesh_data_meshio = meshio.Mesh(
        points=mesh_data["vertices"], cells=[("triangle", mesh_data["triangles"])]
    )

    mesh_scikit = skfem.MeshTri(
        doflocs=mesh_data_meshio.points.T,
        t=mesh_data_meshio.cells_dict["triangle"].T,
    )

    basis_scikit = skfem.Basis(mesh_scikit, skfem.ElementTriP2(), intorder=4)

    @skfem.BilinearForm
    def bilinear_scikit(u, v, _):
        """Stiffness + Mass matrix."""
        return dot(grad(u), grad(v)) + u * v

    @skfem.LinearForm
    def load_scikit(v, w):
        """Load vector."""
        x, y = w.x
        return (2.0 * math.pi**2 * np.sin(math.pi * x) * np.sin(math.pi * y)) * v
        # return x*v

    @skfem.Functional
    def rhs_functional_scikit(w):
        """Integral of the squared rhs."""
        x, y = w.x
        return (2.0 * math.pi**2 * np.sin(math.pi * x) * np.sin(math.pi * y)) ** 2

    stiff_matrix_scikit = torch.tensor(bilinear_scikit.assemble(basis_scikit).todense())
    rhs_vector_scikit = torch.tensor(load_scikit.assemble(basis_scikit)).unsqueeze(-1)
    rhs_functional_scikit_val = torch.tensor(
        rhs_functional_scikit.elemental(basis_scikit)
    ).unsqueeze(-1)

    mesh = MeshTri(mesh_data)

    elements = ElementTri(polynomial_order=2, integration_order=4)

    basis_h = Basis(mesh, elements)

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

    def rhs_functional_form(basis):
        """Integral of the squared rhs."""
        x, y = torch.split(basis.integration_points, 1, dim=-1)

        return rhs(x, y) ** 2

    stiff_matrix = basis_h.integrate_bilinear_form(bilinear)

    rhs_vector = basis_h.integrate_linear_form(residual)

    rhs_functional = basis_h.integrate_functional(rhs_functional_form)

    zero = torch.zeros((1))

    matrix_error_norm = torch.norm(stiff_matrix - stiff_matrix_scikit) / torch.norm(
        stiff_matrix
    )

    rhs_error_norm = torch.norm(rhs_vector - rhs_vector_scikit) / torch.norm(rhs_vector)
    functional_error_norm = torch.norm(
        rhs_functional_scikit_val - rhs_functional
    ) / torch.norm(rhs_functional_scikit_val)

    assert torch.isclose(matrix_error_norm, zero)
    assert torch.isclose(rhs_error_norm, zero)
    assert torch.isclose(functional_error_norm, zero)


if __name__ == "__main__":
    test_assembly()
