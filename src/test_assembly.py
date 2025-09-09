"""Tests for assembly of matrices and vectors."""

import math

import meshio
import numpy as np
import skfem
import torch
import triangle as tr
from skfem.helpers import dot, grad

from tensordict import TensorDict

from fem import Basis, ElementTri, MeshTri


def test_assembly():
    """Test the assembly of matrices and vectors against scikit-fem."""
    torch.set_default_dtype(torch.float64)

    mesh_data = tr.triangulate(
        {"vertices": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]}, "qea0.005"
    )

    mesh_data_meshio = meshio.Mesh(
        points=mesh_data["vertices"], cells=[("triangle", mesh_data["triangles"])]
    )

    mesh_scikit = skfem.MeshTri(
        vertices=mesh_data_meshio.points.T,
        elements=mesh_data_meshio.cells_dict["triangle"].T,
    )

    basis_scikit = skfem.Basis(mesh_scikit, skfem.ElementTriP1(), intorder=3)

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
    rhs_functional_scikit = torch.tensor(
        rhs_functional_scikit.elemental(basis_scikit)
    ).unsqueeze(-1)

    mesh = MeshTri(triangulation=TensorDict(mesh_data))

    elements = ElementTri(polynomial_order=1, integration_order=3)

    basis = Basis(mesh, elements)

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

    stiff_matrix = basis.integrate_bilinear_form(bilinear)

    rhs_vector = basis.integrate_linear_form(residual)

    rhs_functional_scikit = basis.integrate_functional(rhs_functional)

    zero = torch.zeros((1))

    matrix_error_norm = torch.norm(stiff_matrix - stiff_matrix_scikit) / torch.norm(
        stiff_matrix
    )
    rhs_error_norm = torch.norm(rhs_vector - rhs_vector_scikit) / torch.norm(rhs_vector)
    functional_error_norm = torch.norm(
        rhs_functional_scikit - rhs_functional_scikit
    ) / torch.norm(rhs_functional_scikit)

    assert torch.isclose(matrix_error_norm, zero)
    assert torch.isclose(rhs_error_norm, zero)
    assert torch.isclose(functional_error_norm, zero)
