import torch
import math

import matplotlib.pyplot as plt
import triangle as tr

from fem import Fractures, Element_Tri, Basis

# torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
# torch.cuda.empty_cache()
torch.set_default_dtype(torch.float64)

#---------------------- FEM Parameters ----------------------#

h = 0.5**(1)

vertices = [[0., 0.],
            [.5, 0.],
            [1., 0.],
            [.5, .5],
            [0., 1.],
            [.5, 1.],
            [1., 1.]]

vertex_markers = [[1],
                  [1],
                  [1],
                  [2],
                  [1],
                  [1],
                  [1]] 

segments = [[0, 1],
            [0, 4],
            [1, 2],
            [1, 3],
            [2, 6],
            [3, 5],
            [4, 5],
            [5, 6]]

segment_markers =[[1],
                  [1],
                  [1],
                  [2],
                  [1],
                  [2],
                  [1],
                  [1]]

reference_fracture_data = dict(vertices = vertices,
                               vertex_markers = vertex_markers,
                               segments = segments,
                               segment_markers = segment_markers)

reference_fracture_mesh = tr.triangulate(reference_fracture_data, "qpcsea" + str(h))

tr.compare(plt, reference_fracture_data, reference_fracture_mesh)

plt.show()

fractures_data = torch.tensor([[[-1., 0., 0.], 
                                [ 1., 0., 0.], 
                                [-1., 1., 0.], 
                                [ 1., 1., 0.]],
                               [[ 0., 0.,-1.], 
                                [ 0., 0., 1.], 
                                [ 0., 1.,-1.], 
                                [ 0., 1., 1.]]])


mesh = Fractures(triangulation = reference_fracture_mesh,
                 fractures_data = fractures_data)

elements = Element_Tri(P_order = 1, 
                       int_order = 2)

V = Basis(mesh, elements)

#---------------------- Residual Parameters ----------------------#

rhs = lambda x, y : 2. * math.pi**2 * torch.sin(math.pi * x) * torch.sin(math.pi * y)

def l(basis):
    
    x, y = basis.integration_points
            
    v = basis.v
    rhs_value = rhs(x, y)
    
    return rhs_value * v

def a(basis):
    
    v_grad = basis.v_grad
    
    return v_grad @ v_grad.mT

#---------------------- Error Parameters ----------------------#

exact = lambda x, y : torch.sin(math.pi * x) * torch.sin(math.pi * y)
exact_dx = lambda x, y : math.pi * torch.cos(math.pi * x) * torch.sin(math.pi * y)
exact_dy = lambda x, y : math.pi * torch.sin(math.pi * x) * torch.cos(math.pi * y)

def H1_exact(basis):

    x, y = basis.integration_points
    
    return exact(x, y)**2 + exact_dx(x, y)**2 + exact_dy(x, y)**2

def L2_norm(basis, u):
   
    x, y = basis.integration_points
        
    return (exact(x, y)- u)**2
        
exact_norm = torch.sqrt(torch.sum(V.integrate_functional(H1_exact)))

#---------------------- Solution ----------------------#

A = V


#---------------------- Plotting ----------------------#

