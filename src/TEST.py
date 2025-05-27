from skfem import *
from skfem.helpers import dot,grad

mesh = MeshTri().refined(1).with_defaults()
basis = Basis(mesh, ElementTriP2(), intorder = 3, use_torch = True)

@BilinearForm
def a(u,v,w):
    
    return dot(grad(u), grad(v))

@LinearForm
def b(v,w):
    
    # v_grad = w.grad
    
    return v

A = a.assemble(basis)
# b = b.assemble_to_torch(basis)
b = b.assemble(basis)
x = solve(*condense(A, b, D=basis.get_dofs()))

A, b, _, _ = condense(A, b, D=basis.get_dofs())

A = A.todense()
@Functional
def integral(w):
    return w['uh']  # grad, dot, etc. can be used here

uh = basis.interpolate(x)

xd = float(integral.assemble(basis, uh=basis.interpolate(x)))

# print(xd)

import torch

from fem import Mesh, Elements, Basis

import skfem

# torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
# torch.cuda.empty_cache()
# torch.set_default_dtype(torch.float64)

mesh_sk = skfem.MeshTri().refined(1)

coords4nodes = torch.tensor(mesh_sk.p).T

nodes4elements = torch.tensor(mesh_sk.t).T

mesh = Mesh(coords4nodes, nodes4elements)

elements = Elements(P_order = 2, 
                    int_order = 3)

V = Basis(mesh, elements)

def a(elements: Elements):
    
    v = elements.v
    v_grad = elements.v_grad
    
    return v_grad @ v_grad.mT

def l(elements: Elements):
    
    v = elements.v
    
    return 1. * v

def integral(elements, interpolation):
    
    return interpolation

A_xd = V.integrate_bilineal_form(a)[V.inner_dofs, :][:, V.inner_dofs]

b_xd = V.integrate_lineal_form(l)[V.inner_dofs]

solution = torch.linalg.solve(A_xd, b_xd)

sol = torch.zeros(V.nb_global_dofs, 1)

# sol[V.inner_dofs] = solution

# solution_interpolation, _ = V.interpolate_and_grad(sol)

# int_value = V.integrate_functional(integral, solution_interpolation)

# print(int_value.sum().item())

# print(V.elements.inv_mapping_jacobian)
# print(V.elements.inv_map_jacobian)
# print(V.mesh.nodes_idx4boundary_edges.mT)
# print(V.mesh.boundary_edges_idx)

print(V.mesh.nodes4unique_edges.mT)
print(V.mesh.nodes4boundary_edges.mT)