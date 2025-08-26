import torch

import matplotlib.pyplot as plt
import tensordict as td
import triangle as tr

from fracture_fem import Fractures, Element_Fracture, Fracture_Basis

# torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
# torch.cuda.empty_cache()
torch.set_default_dtype(torch.float64)

#---------------------- FEM Parameters ----------------------#

h = 0.5**(10)

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

fractures_triangulation = [fracture_triangulation]

fractures_data = torch.tensor([[[-1., 0., 0.], 
                                [ 1., 0., 0.], 
                                [-1., 1., 0.], 
                                [ 1., 1., 0.]],
                               # [[ 0., 0.,-1.], 
                               #  [ 0., 0., 1.], 
                               #  [ 0., 1.,-1.], 
                               #  [ 0., 1., 1.]]
                               ])

mesh = Fractures(triangulations = fractures_triangulation,
                 fractures_3D_data = fractures_data)

elements = Element_Fracture(P_order = 1, 
                            int_order = 1)

V = Fracture_Basis(mesh, elements)

#---------------------- Residual Parameters ----------------------#

def rhs(x, y, z):
    
    x_fracture_1 = x
    y_fracture_1 = y
    z_fracture_1 = z

    rhs_fracture_1 = 6* x_fracture_1 * (y_fracture_1 - y_fracture_1**2) + 2 *x_fracture_1* (1-x_fracture_1**2)

    rhs_value = rhs_fracture_1
    
    return rhs_value

def l(basis):
    
    x, y, z = basis.integration_points
            
    v = basis.v
    rhs_value = rhs(x, y, z)
    
    return rhs_value * v

def a(basis):
    
    v_grad = basis.v_grad
    
    return v_grad @ v_grad.mT

#---------------------- Error Parameters ----------------------#

exact = lambda x, y, z : y * (1 - y) *x* (1-x**2)
exact_dx = lambda x, y, z : -y * (1 - y) * ((x**2 - 1) + 2 * x * x)
exact_dy = lambda x, y, z : -(1- 2* y) * x * (x**2 - 1)

def H1_exact(basis):
    
    return exact(*basis.integration_points)**2 + exact_dx(*basis.integration_points)**2 + exact_dy(*basis.integration_points)**2

def H1_norm(basis, Ih_u , Ih_u_grad):
       
    Ih_u_dx, Ih_u_dy, _ = torch.split(Ih_u_grad, 1, dim = -1)
    
    return (exact(*basis.integration_points)- Ih_u)**2  + (exact_dx(*basis.integration_points) - Ih_u_dx)**2 + (exact_dy(*basis.integration_points) - Ih_u_dy)**2
        
#---------------------- Solution ----------------------#

A = V.integrate_bilineal_form(a)

b = V.integrate_lineal_form(l)

A_reduced = V.reduce(A)

b_reduced = V.reduce(b)

x = torch.zeros(V.basis_parameters["linear_form_shape"])

x[V.basis_parameters["inner_dofs"]] = torch.linalg.solve(A_reduced, b_reduced)

I_u_h, I_u_h_grad = V.interpolate(x) 

exact_H1_norm = torch.sqrt(torch.sum(V.integrate_functional(H1_exact)))

H1_norm_value = torch.sqrt(torch.sum(V.integrate_functional(H1_norm, I_u_h, I_u_h_grad)))

print((H1_norm_value/exact_H1_norm).item())

#---------------------- Computation Parameters ----------------------#

nb_fractures = mesh.mesh_parameters["nb_fractures"]

local_vertices_3D = V.global_triangulation["vertices_3D"][V.global_triangulation["global2local_idx"].reshape(nb_fractures, -1)] 

exact_value_local = exact(*torch.split(local_vertices_3D, 1, -1)).reshape(-1)

exact_value_global = exact_value_local.reshape(-1,1)[V.global_triangulation["local2global_idx"]].numpy()

x_local = x[V.global_triangulation["global2local_idx"]].reshape(-1)
vertices = mesh.local_triangulations["vertices"].squeeze(0)
triangles = mesh.local_triangulations["triangles"]


#---------------------- Plot ----------------------#

fig = plt.figure(figsize=(10, 4), dpi = 100)

fig.suptitle(r"FEM computed for $F_1$", fontsize=16)

ax1 = fig.add_subplot(1, 3, 1, projection='3d')

ax1.plot_trisurf(vertices[:, 0], 
                 vertices[:, 1], 
                 x_local, 
                 triangles = triangles,
                 cmap='viridis', 
                 edgecolor='black', 
                 linewidth=0.3)

ax1.set_title("FEM solution")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel(r"$u_h(x,y)$")

ax2 = fig.add_subplot(1, 3, 2, projection='3d')

ax2.plot_trisurf(vertices[:, 0], 
                 vertices[:, 1], 
                 exact_value_local, 
                 triangles = triangles,
                 cmap='viridis', 
                 edgecolor='black', 
                 linewidth=0.3)

ax2.set_title("Exact solution")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_zlabel(r"$u(x,y)$")

ax3 = fig.add_subplot(1, 3, 3, projection='3d')

ax3.plot_trisurf(vertices[:, 0], 
                 vertices[:, 1], 
                 abs(exact_value_local - x_local), 
                 triangles = triangles,
                 cmap='viridis', 
                 edgecolor='black', 
                 linewidth=0.3)

ax3.set_title("Error")
ax3.set_xlabel("x")
ax3.set_ylabel("y")
ax3.set_zlabel(r"$|u-u_h|$")


plt.show()

xd_1 = I_u_h