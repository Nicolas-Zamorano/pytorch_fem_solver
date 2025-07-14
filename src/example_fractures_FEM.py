import torch

import matplotlib.pyplot as plt
import tensordict as td
import triangle as tr
import numpy as np

from fracture_fem import Fractures, Element_Fracture, Fracture_Element_Line, Fracture_Basis, Interior_Facet_Fracture_Basis

torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
torch.set_default_dtype(torch.float64)

#---------------------- FEM Parameters ----------------------#

h = 0.5

n = 9

# fracture_2d_data = {"vertices" : [[-1., 0.],
#                                   [ 1., 0.],
#                                   [-1., 1.],
#                                   [ 1., 1.],
#                                   [ 0., 0.],
#                                   [ 0., .5],
#                                   [ 0., 1.]],
#                     "segments" : [[0, 2],
#                                   [0, 4],
#                                   [1, 3],
#                                   [1, 4],
#                                   [2, 6],
#                                   [3, 6],
#                                   [4, 5],
#                                   [5, 6]],
#                     # "segment_markers" : [[2],
#                     #                      [3],
#                     #                      [3],
#                     #                      [3],
#                     #                      [3],
#                     #                      [3],
#                     #                      [0],
#                     #                      [0]]
#                     # "vertex_markers" : [[2],
#                     #                     [3],
#                     #                     [2],
#                     #                     [3],
#                     #                     [0],
#                     #                     [0],
#                     #                     [0]],
#                     }

fracture_2d_data = {"vertices" : [[-1., 0.],
                                  [ 1., 0.],
                                  [-1., 1.],
                                  [ 1., 1.],
                                  [ 0., 0.],
                                  [ 0., 1.]],
                    "segments" : [[0, 2],
                                  [0, 4],
                                  [1, 3],
                                  [1, 4],
                                  [2, 5],
                                  [3, 5],
                                  [4, 5]],
                    "segment_markers": [[1],
                                        [1],
                                        [1],
                                        [1],
                                        [1],
                                        [1],
                                        [0]]
                    }

fracture_triangulation = tr.triangulate(fracture_2d_data, 
                                                      "pqsea"+str(h**n)
                                                      )

fracture_triangulation_torch = td.TensorDict(fracture_triangulation)

# tr.compare(plt, fracture_2d_data, fracture_triangulation)

# plt.show()

fractures_triangulation = (fracture_triangulation_torch, fracture_triangulation_torch)



fractures_data = torch.tensor([[[-1., 0., 0.], 
                                [ 1., 0., 0.], 
                                [-1., 1., 0.], 
                                [ 1., 1., 0.]],
                               
                               [[ 0., 0.,-1.], 
                                [ 0., 0., 1.], 
                                [ 0., 1.,-1.], 
                                [ 0., 1., 1.]]])

mesh = Fractures(triangulations = fractures_triangulation,
                 fractures_3D_data = fractures_data)

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
    
    # rhs_fracture_1 = torch.ones_like(x_fracture_1) 
    # rhs_fracture_2 = torch.zeros_like(x_fracture_1) 
    # rhs_fracture_2 = torch.ones_like(x_fracture_1) 

    # rhs_fracture_1 = - 3 * (x_fracture_1 - 1) * y_fracture_1 * (1-y_fracture_1) + 2*x_fracture_1*(0.5-x_fracture_1)*(1-x_fracture_2)
    # rhs_fracture_1 = 6* x_fracture_1 * (y_fracture_1 - y_fracture_1**2) + 2 *x_fracture_1* (1-x_fracture_1**2)
    
    # rhs_fracture_1 = torch.zeros_like(x_fracture_1) 
    # rhs_fracture_2 = torch.zeros_like(x_fracture_2)
    
    rhs_value = torch.cat([rhs_fracture_1, rhs_fracture_2], dim = 0)

    return rhs_value

def l(basis):
    
    x, y, z = basis.integration_points
            
    v = basis.v
    rhs_value = rhs(x, y, z)
    
    return rhs_value * v

def a(basis):
    
    v_grad = basis.v_grad
    
    return v_grad @ v_grad.mT

# def g(x, y):
#     return torch.ones_like(x)

# def g(x, y):
#     return x + torch.sqrt(y)

#---------------------- Error Parameters ----------------------#

def exact(x, y, z):
    
    x_fracture_1, x_fracture_2 = torch.split(x, 1, dim = 0)
    y_fracture_1, y_fracture_2 = torch.split(y, 1, dim = 0)
    z_fracture_1, z_fracture_2 = torch.split(z, 1, dim = 0)

    exact_fracture_1 = -y_fracture_1 * (1 - y_fracture_1) * torch.abs(x_fracture_1) * (x_fracture_1**2 - 1)
    exact_fracture_2 =  y_fracture_2 * (1 - y_fracture_2) * torch.abs(z_fracture_2) * (z_fracture_2**2 - 1)

    
    exact_value = torch.cat([exact_fracture_1, exact_fracture_2], dim = 0)

    return exact_value

def exact_grad(x,y,z):
    
    x_fracture_1, x_fracture_2 = torch.split(x, 1, dim = 0)
    y_fracture_1, y_fracture_2 = torch.split(y, 1, dim = 0)
    z_fracture_1, z_fracture_2 = torch.split(z, 1, dim = 0)
    
    exact_dx_fracture_1 = -y_fracture_1 * (1 - y_fracture_1) * (torch.sign(x_fracture_1) * (x_fracture_1**2 - 1) + 2 * x_fracture_1 * torch.abs(x_fracture_1))
    exact_dy_fracture_1 = -(1- 2* y_fracture_1) * torch.abs(x_fracture_1) * (x_fracture_1**2 - 1)
    exact_dz_fracture_1 = torch.zeros_like(exact_dx_fracture_1)
    
    exact_grad_fracture_1 = torch.cat([exact_dx_fracture_1, exact_dy_fracture_1, exact_dz_fracture_1], dim = -1)
    
    exact_dy_fracture_2 = (1- 2* y_fracture_2) * torch.abs(z_fracture_2) * (z_fracture_2**2 - 1)
    exact_dz_fracture_2 = y_fracture_2 * (1 - y_fracture_2) * (torch.sign(z_fracture_2) * (z_fracture_2**2 - 1) + 2 * z_fracture_2 * torch.abs(z_fracture_2))
    exact_dx_fracture_2 = torch.zeros_like(exact_dz_fracture_2)

    exact_grad_fracture_2 = torch.cat([exact_dx_fracture_2, exact_dy_fracture_2, exact_dz_fracture_2], dim = -1)
    
    grad_value = torch.cat([exact_grad_fracture_1, exact_grad_fracture_2], dim = 0)

    return grad_value

def H1_exact(basis):
    
    exact_value = exact(*basis.integration_points)
    
    exact_dx_value, exact_dy_value, exact_dz_value = torch.split(exact_grad(*basis.integration_points), 1, dim = -1)
    
    return exact_value**2 + exact_dx_value**2 + exact_dy_value**2 + exact_dz_value**2

def H1_norm(basis, I_u_h, I_u_h_grad):
    
    exact_value = exact(*basis.integration_points)

    exact_dx_value, exact_dy_value, exact_dz_value = torch.split(exact_grad(*basis.integration_points), 1, dim = -1)
    
    Ih_x_dx, Ih_x_dy, Ih_x_dz = torch.split(I_u_h_grad, 1, dim = -1)
    
    L2_error = (exact_value - I_u_h)**2
           
    H1_0_error = (exact_dx_value - Ih_x_dx)**2 + (exact_dy_value - Ih_x_dy)**2 + (exact_dz_value - Ih_x_dz)**2
    
    return H1_0_error + L2_error
            
exact_norm = torch.sqrt(V.integrate_functional(H1_exact))

#---------------------- Solution ----------------------#

A = V.integrate_bilineal_form(a)

b = V.integrate_lineal_form(l)

A_reduced = V.reduce(A)

b_reduced = V.reduce(b)

u_h = torch.zeros(V.basis_parameters["linear_form_shape"])

u_h[V.basis_parameters["inner_dofs"]] = torch.linalg.solve(A_reduced, b_reduced)

I_u_h, I_u_h_grad = V.interpolate(V, u_h) 

# non_zero_dirichlet_dofs = torch.nonzero(V.global_triangulation["vertex_markers"] == 2)[:, 0] 

# A_tilde, b_tilde = V.apply_BC(A, b, g, non_zero_dirichlet_dofs) 

# u_h = torch.zeros(V.basis_parameters["linear_form_shape"])

# A_reduced = V.reduce(A_tilde)

# b_reduced = V.reduce(b_tilde)

# u_h[V.basis_parameters["inner_dofs"]] = torch.linalg.solve(A_reduced, b_reduced)

# I_u_h, I_u_h_grad = V.interpolate(u_h) 

#---------------------- Plot Values ----------------------#

### --- FEM SOLUTION PARAMETERS --- ###

vertices_fracture_1, vertices_fracture_2 = torch.unbind(mesh.local_triangulations["vertices"], dim = 0)

triangles_fracture_1, triangles_fracture_2 =  torch.unbind(mesh.local_triangulations["triangles"], dim = 0)

u_h_fracture_1 , u_h_fracture_2 = torch.unbind(u_h[V.global_triangulation["global2local_idx"]].reshape(2, -1, 1), dim = 0)

### --- TRACE PARAMETERS --- ###

V_inner_edges = Interior_Facet_Fracture_Basis(mesh, Fracture_Element_Line(P_order = 1, int_order = 2))

traces_local_edges_idx = V.global_triangulation["traces_local_edges_idx"]

n_E = V.mesh.local_triangulations["normal4inner_edges_3D"]

n4e_u =  mesh.edges_parameters["nodes4unique_edges"]

nodes4trace = n4e_u[torch.arange(n4e_u.shape[0])[:, None], traces_local_edges_idx]

c4n = V.mesh.local_triangulations["vertices"]

coords4trace = c4n[torch.arange(c4n.shape[0]), nodes4trace]

# points_trace_fracture_1, points_trace_fracture_2 = torch.unbind(, dim = 0)

sort_points_trace, sort_idx = torch.sort(coords4trace.mean(-2)[...,1])

points_trace_fracture_1, points_trace_fracture_2 = torch.unbind(sort_points_trace, dim = 0)

# COMPUTE JUMP FEM SOLUTION

I_E_u_h, I_E_u_grad = V.interpolate(V_inner_edges, u_h)

I_E_u_h_grad_K_plus, I_E_u_h_grad_minus = torch.unbind(I_E_u_grad, dim = -4)

jump_u_h = (I_E_u_h_grad_K_plus * n_E).sum(-1) + (I_E_u_h_grad_minus * -n_E).sum(-1)

jump_u_h_trace = jump_u_h[torch.arange(jump_u_h.shape[0])[:, None], traces_local_edges_idx]

sort_jump_u_h_trace = jump_u_h_trace[torch.arange(jump_u_h_trace.shape[0])[:, None], sort_idx]

jump_u_h_trace_fracture_1, jump_u_h_trace_fracture_2 = torch.unbind(sort_jump_u_h_trace, dim = 0)

# COMPUTE JUMP EXACT 

local_vertices_3D = V.global_triangulation["vertices_3D"][V.global_triangulation["global2local_idx"].reshape(2, -1)] 

exact_value_local = exact(*torch.split(local_vertices_3D, 1, -1))

exact_value_global = exact_value_local.reshape(-1,1)[V.global_triangulation["local2global_idx"]]

I_E_u, I_E_u_grad = V.interpolate(V_inner_edges, exact_value_global)

I_E_u_grad_K_plus, I_E_u_grad_minus = torch.unbind(I_E_u_grad, dim = -4)

n_E = V.mesh.local_triangulations["normal4inner_edges_3D"]

jump_u = (I_E_u_grad_K_plus * n_E).sum(-1) + (I_E_u_grad_minus * -n_E).sum(-1)

jump_u_trace = jump_u[torch.arange(jump_u.shape[0])[:, None], traces_local_edges_idx]

sort_jump_u_trace = jump_u_trace[torch.arange(jump_u_trace.shape[0])[:, None], sort_idx]

jump_u_trace_fracture_1, jump_u_trace_fracture_2 = torch.unbind(sort_jump_u_trace, dim = 0)

### --- ERROR PARAMETERS --- ###

H1_error_fracture_1, H1_error_fracture_2 = torch.unbind(torch.sqrt(V.integrate_functional(H1_norm, I_u_h, I_u_h_grad)/V.integrate_functional(H1_exact)), dim = 0)

c4e_fracture_1, c4e_fracture_2 =  torch.unbind(mesh.local_triangulations["coords4triangles"], dim = 0)

#---------------------- Plot ----------------------#

### --- PLOT FEM SOLUTION --- ###

fig_sol = plt.figure(figsize = (10, 4), dpi = 200)
fig_sol.suptitle("FEM solution", fontsize = 16)

ax_fracture_1 = fig_sol.add_subplot(1, 2, 1, projection = '3d')
ax_fracture_1.plot_trisurf(vertices_fracture_1.numpy(force = True)[:, 0], 
                 vertices_fracture_1.numpy(force = True)[:, 1], 
                 u_h_fracture_1.reshape(-1).numpy(force = True), 
                 triangles = triangles_fracture_1.numpy(force = True),
                 cmap = 'viridis', 
                 edgecolor = 'black', 
                 linewidth = 0.1)

ax_fracture_1.set_title("Fracture 1")
ax_fracture_1.set_xlabel(r"$x$")
ax_fracture_1.set_ylabel(r"$y$")
ax_fracture_1.set_zlabel(r"$u_h(x,y)$")

ax_fracture_2 = fig_sol.add_subplot(1, 2, 2, projection = '3d')
ax_fracture_2.plot_trisurf(vertices_fracture_2.numpy(force = True)[:, 0], 
                 vertices_fracture_2.numpy(force = True)[:, 1], 
                 u_h_fracture_2.reshape(-1).numpy(force = True),
                 triangles = triangles_fracture_2.numpy(force = True),
                 cmap = 'viridis', 
                 edgecolor = 'black', 
                 linewidth = 0.1)

ax_fracture_2.set_title("Fracture 2")
ax_fracture_2.set_xlabel(r"$x$")
ax_fracture_2.set_ylabel(r"$y$")
ax_fracture_2.set_zlabel(r"$u_h(x,y)$")

# plt.subplots_adjust(wspace = 0.4)  # Aumenta separación horizontal
plt.show()

### --- PLOT TRACES --- ###

fig_traces = plt.figure(figsize = (11, 4), dpi = 200)
fig_traces.suptitle("Traces values for FEM solution", fontsize = 16)

ax_jump_fracture_1 = fig_traces.add_subplot(1, 2, 1)
ax_jump_fracture_1.plot(points_trace_fracture_1.numpy(force = True),
                        jump_u_trace_fracture_1.reshape(-1).numpy(force = True),
                        color = "black",
                        label = r"$u^{ex}$")
ax_jump_fracture_1.scatter(points_trace_fracture_1.numpy(force = True),
                           jump_u_h_trace_fracture_1.reshape(-1).numpy(force = True),
                           color = "r",
                           label = r"$u_h$")
ax_jump_fracture_1.set_title("Fracture 1")
ax_jump_fracture_1.set_xlabel("trace lenght")
ax_jump_fracture_1.set_ylabel("jump value")
ax_jump_fracture_1.legend()

ax_jump_fracture_2 = fig_traces.add_subplot(1, 2, 2)
ax_jump_fracture_2.plot(points_trace_fracture_2.numpy(force = True),
                        jump_u_trace_fracture_2.reshape(-1).numpy(force = True),
                        color = "black",
                        label = r"$u^{ex}$")
ax_jump_fracture_2.scatter(points_trace_fracture_2.numpy(force = True),
                           jump_u_h_trace_fracture_2.reshape(-1).numpy(force = True),
                           color = "r",
                           label = r"$u_h$")
ax_jump_fracture_2.set_title("Fracture 2")
ax_jump_fracture_2.set_xlabel("trace lenght")
ax_jump_fracture_2.set_ylabel("jump value")
ax_jump_fracture_2.legend()

plt.show()

### --- PLOT ERROR --- ###

from matplotlib.collections import PolyCollection
import matplotlib.cm as cm
import matplotlib.colors as colors

# Obtener los errores y coordenadas
H1_error_fracture_1 = H1_error_fracture_1.numpy(force = True)
c4e_fracture_1 = c4e_fracture_1.numpy(force = True)

H1_error_fracture_2 = H1_error_fracture_2.numpy(force = True)
c4e_fracture_2 = c4e_fracture_2.numpy(force = True)

# Unificar escala de colores
all_errors = np.concatenate([H1_error_fracture_1, H1_error_fracture_2])
norm = colors.Normalize(vmin = all_errors.min(), vmax = all_errors.max())
cmap = cm.viridis

# Crear figura con 2 subplots
fig_error, axes = plt.subplots(1,2, 
                               figsize = (12, 3), 
                               dpi = 200)

fig_error.suptitle("Relative error for FEM solution", fontsize = 14)

# Fracture 1
face_colors_1 = cmap(norm(H1_error_fracture_1))
collection1 = PolyCollection(c4e_fracture_1, 
                             facecolors = face_colors_1, 
                             edgecolors = 'black', 
                             linewidths = 0.2)
ax1 = axes[0]
ax1.add_collection(collection1)
ax1.autoscale()
ax1.set_aspect('equal')
ax1.set_title('Fracture 1')
ax1.set_xlim([-1, 1])
ax1.set_ylim([0, 1])

# Fracture 2
face_colors_2 = cmap(norm(H1_error_fracture_2))
collection2 = PolyCollection(c4e_fracture_2, 
                             facecolors = face_colors_2, 
                             edgecolors = 'black', 
                             linewidths = 0.2)
ax2 = axes[1]
ax2.add_collection(collection2)
ax2.autoscale()
ax2.set_aspect('equal')
ax2.set_title('Fracture 2')
ax2.set_xlim([-1, 1])
ax2.set_ylim([0, 1])

# Colorbar común
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array(all_errors)
cbar = fig_error.colorbar(sm, 
                          ax = axes.ravel().tolist(), 
                          orientation = 'vertical', 
                          label = 'error')

plt.show()