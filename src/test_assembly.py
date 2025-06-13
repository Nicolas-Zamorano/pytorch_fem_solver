import meshio
import skfem
import torch
import math

from fem import Mesh_Tri, Element_Tri, Basis
from skfem.helpers import dot, grad 

import triangle as tr
import numpy as np

mesh_data = tr.triangulate({"vertices":[[0., 0.],
                                        [1., 0.],
                                        [0., 1.],
                                        [1., 1.]]}, 
                            "qea0.005")

##------------------------ scikit-fem ------------------------##

meshio.Mesh(points = mesh_data["vertices"],
            cells = [("triangle", mesh_data["triangles"])]).write("mesh.msh")

mesh_sk = skfem.MeshTri().load("mesh.msh")

V_sk = skfem.Basis(mesh_sk, skfem.ElementTriP1(), intorder = 3)

@skfem.BilinearForm
def a(u,v,_):
    return dot(grad(u), grad(v)) + u * v

@skfem.LinearForm
def l(v, w):
    x,y = w.x
    return( 2. * math.pi**2 * np.sin(math.pi * x) * np.sin(math.pi * y)) * v
    # return x*v
    
@skfem.Functional
def rhs_int(w):
    x,y = w.x
    return ( 2. * math.pi**2 * np.sin(math.pi * x) * np.sin(math.pi * y))**2

A_sk = torch.tensor(a.assemble(V_sk).todense())
b_sk = torch.tensor(l.assemble(V_sk)).unsqueeze(-1)
rhs_int_sk = torch.tensor(rhs_int.elemental(V_sk)).unsqueeze(-1)

##------------------------ scikit-fem ------------------------##

mesh = Mesh_Tri(triangulation = mesh_data)

elements = Element_Tri(P_order = 1, 
                       int_order = 3)

V = Basis(mesh, elements)

def gram_matrix(elements):
    
    v = elements.v
    v_grad = elements.v_grad
    
    return v_grad @ v_grad.mT + v @ v.mT

rhs = lambda x, y: 2. * math.pi**2 * torch.sin(math.pi * x) * torch.sin(math.pi * y)

def residual(elements):
    
    x, y = elements.integration_points
    v = elements.v
    
    return rhs(x,y) * v 

def functional(elements):
    
    x, y = elements.integration_points
    
    return rhs(x,y)**2

A = V.integrate_bilineal_form(gram_matrix)

b = V.integrate_lineal_form(residual)

rhs_int = V.integrate_functional(functional)

##------------------------ test ------------------------##

assert np.isclose(torch.norm(A - A_sk)/torch.norm(A), 0.)
assert np.isclose(torch.norm(b - b_sk)/torch.norm(b), 0.)
assert np.isclose(torch.norm(rhs_int - rhs_int_sk)/torch.norm(rhs_int_sk), 0.)