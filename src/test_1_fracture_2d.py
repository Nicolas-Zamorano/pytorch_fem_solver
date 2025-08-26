import torch

import matplotlib.pyplot as plt
import tensordict as td
import triangle as tr

from fem import Mesh_Tri, Element_Tri, Basis

torch.set_default_dtype(torch.float64)

rhs = lambda x, y : 6* x * (y - y**2) + 2 *x* (1-x**2)

def l(basis):
    
    x, y = basis.integration_points
            
    v = basis.v
    rhs_value = rhs(x, y)
    
    return rhs_value * v

def a(basis):
    
    v_grad = basis.v_grad
    
    return v_grad @ v_grad.mT

h = 0.5**10

exact = lambda x, y : y * (1 - y) *x* (1-x**2)
exact_dx = lambda x, y : -y * (1 - y) * ((x**2 - 1) + 2 * x * x)
exact_dy = lambda x, y : -(1- 2* y) * x * (x**2 - 1)

def H1_exact(basis):

    x, y = basis.integration_points
    
    return exact(x, y)**2 + exact_dx(x, y)**2 + exact_dy(x,y)**2

def H1_norm(basis, Ih_u , Ih_u_grad):
   
    x, y = basis.integration_points
    
    Ih_u_dx, Ih_u_dy = torch.split(Ih_u_grad, 1, dim = -1)
    
    return (exact(x, y)- Ih_u)**2  + (exact_dx(x, y) - Ih_u_dx)**2 + (exact_dy(x, y) - Ih_u_dy)**2
        
fracture_2d_data = {"vertices" : [[-1., 0.],
                                  [ 1., 0.],
                                  [-1., 1.],
                                  [ 1., 1.],
                                  [ 0., 0.],
                                  [ 0., .5],
                                  [ 0., 1.]],
                    "segments" : [[0, 1],
                                  [1, 3],
                                  [2, 3],
                                  [0, 2],
                                  [4, 5],
                                  [5, 6]]
                    }

fracture_triangulation = td.TensorDict(tr.triangulate(fracture_2d_data, 
                                                      "pqsena"+str(h)
                                                      ))

mesh = Mesh_Tri(triangulation = fracture_triangulation)

elements = Element_Tri(P_order = 1, 
                       int_order = 1)

V = Basis(mesh, elements)

A = V.integrate_bilineal_form(a)

b = V.integrate_lineal_form(l)

A_reduced = V.reduce(A)

b_reduced = V.reduce(b)

x = torch.zeros(V.basis_parameters["linear_form_shape"])

x[V.basis_parameters["inner_dofs"]] = torch.linalg.solve(A_reduced, b_reduced)

exact_value = exact(*torch.unbind(mesh.coords4nodes , -1))
rhs_value = rhs(*torch.unbind(mesh.coords4nodes , -1))

I_u_h, I_u_h_grad = V.interpolate(V, x) 

exact_H1_norm = torch.sqrt(torch.sum(V.integrate_functional(H1_exact)))

H1_norm_value = torch.sqrt(torch.sum(V.integrate_functional(H1_norm, I_u_h, I_u_h_grad)))

print((H1_norm_value/exact_H1_norm).item())

# fig = plt.figure(figsize=(10, 4), dpi = 100)
# fig.suptitle(r"FEM computed for $\omega_1$", fontsize=16)

# ax1 = fig.add_subplot(1, 3, 1, projection='3d')

# ax1.plot_trisurf(mesh.coords4nodes[:, 0], 
#                  mesh.coords4nodes[:, 1], 
#                  x.squeeze(-1), 
#                  triangles = mesh.nodes4elements,
#                  cmap='viridis', 
#                  edgecolor='black', 
#                  linewidth=0.3)

# ax1.set_title("FEM solution")
# ax1.set_xlabel("x")
# ax1.set_ylabel("y")
# ax1.set_zlabel(r"$u_h(x,y)$")

# ax2 = fig.add_subplot(1, 3, 2, projection='3d')

# ax2.plot_trisurf(mesh.coords4nodes[:, 0], 
#                  mesh.coords4nodes[:, 1], 
#                  exact_value, 
#                  triangles = mesh.nodes4elements,
#                  cmap='viridis', 
#                  edgecolor='black', 
#                  linewidth=0.3)

# ax2.set_title("Exact solution")
# ax2.set_xlabel("x")
# ax2.set_ylabel("y")
# ax2.set_zlabel(r"$u(x,y)$")

# ax3 = fig.add_subplot(1, 3, 3, projection='3d')

# ax3.plot_trisurf(mesh.coords4nodes[:, 0], 
#                  mesh.coords4nodes[:, 1], 
#                  abs(exact_value - x.squeeze(-1)), 
#                  triangles = mesh.nodes4elements,
#                  cmap='viridis', 
#                  edgecolor='black', 
#                  linewidth=0.3)

# ax3.set_title("Error")
# ax3.set_xlabel("x")
# ax3.set_ylabel("y")
# ax3.set_zlabel(r"$|u-u_h|$")

# plt.show()

xd_2 = I_u_h
