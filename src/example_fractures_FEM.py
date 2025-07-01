import torch

import matplotlib.pyplot as plt
import tensordict as td
import triangle as tr
import pyvista as pv
import numpy as np

from fem import Fractures, Element_Fracture, Fracture_Basis

# torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
# torch.cuda.empty_cache()
torch.set_default_dtype(torch.float64)

#---------------------- FEM Parameters ----------------------#

plot_mesh = True

h = 0.5**(4)

vertices = [[0., 0.],
            [.5, 0.],
            [1., 0.],
            [.5, .5],
            [0., 1.],
            [.5, 1.],
            [1., 1.]]

# vertices = [[-1., 0.],
#             [0, 0.],
#             [1., 0.],
#             [0, .5],
#             [-1., 1.],
#             [0, 1.],
#             [1., 1.]]

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

fracture_1_data = dict(vertices = vertices,
                       vertex_markers = vertex_markers,
                       segments = segments,
                       segment_markers = segment_markers)

fracture_1_mesh = tr.triangulate(fracture_1_data, "qpcsea" + str(h))

tr.compare(plt, fracture_1_data, fracture_1_mesh)
plt.show()

fracture_1_mesh = td.TensorDict(fracture_1_mesh)

fractures_list =  (fracture_1_mesh, fracture_1_mesh)

fractures_data = torch.tensor([[[-1., 0., 0.], 
                                [ 1., 0., 0.], 
                                [-1., 1., 0.], 
                                [ 1., 1., 0.]],
                               
                               [[ 0., 0.,-1.], 
                                [ 0., 0., 1.], 
                                [ 0., 1.,-1.], 
                                [ 0., 1., 1.]]])

mesh = Fractures(triangulations = fractures_list,
                 fractures_data = fractures_data)

elements = Element_Fracture(P_order = 1, 
                            int_order = 4)

V = Fracture_Basis(mesh, elements)

#---------------------- Residual Parameters ----------------------#

def rhs(x, y, z):
    
    x_fracture_1, x_fracture_2 = torch.split(x, 1, dim = 0)
    y_fracture_1, y_fracture_2 = torch.split(y, 1, dim = 0)
    z_fracture_1, z_fracture_2 = torch.split(z, 1, dim = 0)

    rhs_fracture_1 =  6. * (y_fracture_1 - y_fracture_1**2) * torch.abs(x_fracture_1) - 2. * (torch.abs(x_fracture_1)**3 - torch.abs(x_fracture_1))
    rhs_fracture_2 = -6. * (y_fracture_2 - y_fracture_2**2) * torch.abs(z_fracture_2) + 2. * (torch.abs(z_fracture_2)**3 - torch.abs(z_fracture_2))
    
    rhs_value = torch.cat([rhs_fracture_1, rhs_fracture_2], dim = 0)

    return rhs_value

# rhs = lambda x,y,z: torch.ones_like(x)

def l(basis):
    
    x, y, z = basis.integration_points
            
    v = basis.v
    rhs_value = rhs(x, y, z)
    
    return 2 * rhs_value * v

def a(basis):
    
    v_grad = basis.v_grad
    
    return v_grad @ v_grad.mT

#---------------------- Error Parameters ----------------------#

def exact(x, y, z):
    
    x_fracture_1, x_fracture_2 = torch.split(x, 1, dim = 0)
    y_fracture_1, y_fracture_2 = torch.split(y, 1, dim = 0)
    z_fracture_1, z_fracture_2 = torch.split(z, 1, dim = 0)

    exact_fracture_1 = -y_fracture_1 * (1 - y_fracture_1) * torch.abs(x_fracture_1) * (x_fracture_1**2 - x_fracture_1)
    exact_fracture_2 =  y_fracture_2 * (1 - y_fracture_2) * torch.abs(z_fracture_2) * (z_fracture_2**2 - z_fracture_2)
    
    exact_value = torch.cat([exact_fracture_2, exact_fracture_1], dim = 0)

    return exact_value

def L2_exact(basis):
    
    return exact(*basis.integration_points)**2

def L2_norm(basis, u):
           
    return (exact(*basis.integration_points)- u)**2
        
exact_norm = torch.sqrt(torch.sum(V.integrate_functional(L2_exact)))

#---------------------- Solution ----------------------#3

A = V.integrate_bilineal_form(a)

plt.spy(A.numpy())

plt.show()

b = V.integrate_lineal_form(l)

A_reduced = V.reduce(A)


#---------------------- Plotting ----------------------#

