import torch

import matplotlib.pyplot as plt
import tensordict as td
import triangle as tr

from Neural_Network import Neural_Network_3D
from fracture_fem import Fractures, Element_Fracture, Fracture_Element_Line, Fracture_Basis, Interior_Facet_Fracture_Basis
from datetime import datetime
import numpy as np

torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
torch.set_default_dtype(torch.float64)

#---------------------- Neural Network Functions ----------------------#

def NN_gradiant(NN, x, y, z):

    x.requires_grad_(True)
    y.requires_grad_(True)
    z.requires_grad_(True)

    output = NN(x, y, z)
    
    gradients = torch.autograd.grad(outputs = output,
                                    inputs = (x, y, z),
                                    grad_outputs = torch.ones_like(output),
                                    retain_graph = True,
                                    create_graph = True)
        
    return torch.concat(gradients, dim = -1)

def optimizer_step(optimizer, loss_value):
        optimizer.zero_grad()
        loss_value.backward(retain_graph = True)
        optimizer.step()
        scheduler.step()

#---------------------- Neural Network Parameters ----------------------#

epochs = 5000
learning_rate = 0.2e-3
decay_rate = 0.98
decay_steps = 200

NN = torch.jit.script(Neural_Network_3D(input_dimension = 3, 
                                        output_dimension = 1,
                                        deep_layers = 4, 
                                        hidden_layers_dimension = 15,
                                        activation_function= torch.nn.ReLU()))

optimizer = torch.optim.Adam(NN.parameters(), 
                             lr = learning_rate)  

scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 
                                                   decay_rate ** (1/decay_steps))

#---------------------- FEM Parameters ----------------------#

h = 0.5

n = 10

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
                    }

fracture_triangulation = tr.triangulate(fracture_2d_data, 
                                                      "pqsea"+str(h**n)
                                                      )

fracture_triangulation_torch = td.TensorDict(fracture_triangulation)

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
                            int_order = 2)

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

def residual(basis, u_NN_grad):
        
    NN_grad = u_NN_grad(*basis.integration_points)
        
    v = basis.v
    v_grad = basis.v_grad
    rhs_value = rhs(*basis.integration_points)
    
    return rhs_value * v - v_grad @ NN_grad.mT

def gram_matrix(basis):
    
    v_grad = basis.v_grad
    
    return v_grad @ v_grad.mT

A = V.reduce(V.integrate_bilineal_form(gram_matrix))

A_inv = torch.linalg.inv(A)

Ih, Ih_grad = V.interpolate(V)

Ih_NN = lambda x, y, z : Ih(NN)
Ih_grad_NN = lambda x, y, z : Ih_grad(NN)

# Ih_NN = lambda x, y, z : NN(x, y, z)
# Ih_grad_NN = lambda x, y, z : NN_gradiant(NN, x, y, z)

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

    # return exact_value**2 

def H1_norm(basis, u_NN, u_NN_grad):
    
    exact_value = exact(*basis.integration_points)
    
    u_NN_value = u_NN(*basis.integration_points)
    
    L2_error = (exact_value - u_NN_value)**2

    exact_dx_value, exact_dy_value, exact_dz_value = torch.split(exact_grad(*basis.integration_points), 1, dim = -1)
    
    u_NN_dx, u_NN_dy, u_NN_dz = torch.split(u_NN_grad(*basis.integration_points), 1, dim = -1)
           
    H1_0_error = (exact_dx_value - u_NN_dx)**2 + (exact_dy_value - u_NN_dy)**2 + (exact_dz_value - u_NN_dz)**2
    
    return H1_0_error + L2_error

    # return L2_error
            
exact_norm = torch.sqrt(torch.sum(V.integrate_functional(H1_exact)))

loss_list = []
relative_loss_list = []
H1_error_list = []

loss_opt = 10e4

#---------------------- Training ----------------------#

start_time = datetime.now()

for epoch in range(epochs):
    current_time = datetime.now().strftime("%H:%M:%S")
    print(f"{'='*20} [{current_time}] Epoch:{epoch + 1}/{epochs} {'='*20}")
    
    residual_value = V.reduce(V.integrate_lineal_form(residual, Ih_grad_NN))
                                          
    loss_value = residual_value.T @ (A_inv @ residual_value)
    
    # loss_value = (residual_value**2).sum()

    optimizer_step(optimizer, loss_value)
    
    error_norm = torch.sqrt(torch.sum(V.integrate_functional(H1_norm, Ih_NN, Ih_grad_NN)))/exact_norm

    relative_loss = torch.sqrt(loss_value)/exact_norm 
            
    print(f"Loss: {loss_value.item():.8f} Relative Loss: {relative_loss.item():.8f} Relative error: {error_norm.item():.8f}")
        
    if loss_value < loss_opt:
        loss_opt = loss_value
        params_opt = NN.state_dict()

    loss_list.append(loss_value.item())
    relative_loss_list.append(relative_loss.item())
    H1_error_list.append(error_norm.item())

end_time = datetime.now()

execution_time = end_time - start_time

print(f"Training time: {execution_time}")

#---------------------- Plot Values ----------------------#

NN.load_state_dict(params_opt)

### --- FEM SOLUTION PARAMETERS --- ###

local_vertices_3D = V.global_triangulation["vertices_3D"][V.global_triangulation["global2local_idx"].reshape(2, -1)] 

vertices_fracture_1, vertices_fracture_2 = torch.unbind(mesh.local_triangulations["vertices"], dim = 0)

triangles_fracture_1, triangles_fracture_2 =  torch.unbind(mesh.local_triangulations["triangles"], dim = 0)

u_NN_local = NN(*torch.split(local_vertices_3D, 1, -1))

u_NN_fracture_1 , u_NN_fracture_2 = torch.unbind(u_NN_local, dim = 0)


### --- TRACE PARAMETERS --- ###

trace_nodes = V.global_triangulation["vertices_3D"][V.global_triangulation["traces__global_vertices_idx"], 1].numpy(force = True)

local_vertices_3D = V.global_triangulation["vertices_3D"][V.global_triangulation["global2local_idx"].reshape(2, -1)] 

exact_value_local = exact(*torch.split(local_vertices_3D, 1, -1))

exact_value_global = exact_value_local.reshape(-1,1)[V.global_triangulation["local2global_idx"]]

exact_trace = exact_value_global[V.global_triangulation["traces__global_vertices_idx"]].numpy(force = True)

u_NN_global = u_NN_local.reshape(-1,1)[V.global_triangulation["local2global_idx"]]

u_NN_trace = u_NN_global[V.global_triangulation["traces__global_vertices_idx"]].numpy(force = True)

### --- JUMP PARAMETERS --- ###

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

# COMPUTE JUMP NN SOLUTION

_, I_E_grad = V.interpolate(V_inner_edges)

I_E_NN_grad = I_E_grad(NN)

I_E_u_NN_grad_K_plus, I_E_u_NN_grad_minus = torch.unbind(I_E_NN_grad, dim = -4)

jump_u_NN = (I_E_u_NN_grad_K_plus * n_E).sum(-1) + (I_E_u_NN_grad_minus * -n_E).sum(-1)

jump_u_NN_trace = jump_u_NN[torch.arange(jump_u_NN.shape[0])[:, None], traces_local_edges_idx]

sort_jump_u_NN_trace = jump_u_NN_trace[torch.arange(jump_u_NN_trace.shape[0])[:, None], sort_idx]

jump_u_NN_trace_fracture_1, jump_u_NN_trace_fracture_2 = torch.unbind(sort_jump_u_NN_trace, dim = 0)

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

numerator = V.integrate_functional(H1_norm, Ih_NN, Ih_grad_NN)
denominator = V.integrate_functional(H1_exact)

epsilon = 1e-10
safe_denominator = torch.where(denominator.abs() < epsilon, torch.ones_like(denominator), denominator)

H1_error_fracture_1, H1_error_fracture_2 = torch.unbind(
    torch.sqrt(numerator / safe_denominator),
    dim=0
)

# H1_error_fracture_1, H1_error_fracture_2 =  torch.unbind(torch.sqrt(V.integrate_functional(H1_norm, Ih_NN, Ih_grad_NN)/V.integrate_functional(H1_exact)), dim = 0)

print(torch.sqrt(torch.sum(V.integrate_functional(H1_norm, Ih_NN, Ih_grad_NN)))/exact_norm)

c4e_fracture_1, c4e_fracture_2 =  torch.unbind(mesh.local_triangulations["coords4triangles"], dim = 0)

# ---------------------- Plot ---------------------- #

import pyvista as pv
from matplotlib.collections import PolyCollection
import matplotlib.cm as cm
import matplotlib.colors as colors
import os

# ------------------ CONFIGURACIÓN DE GUARDADO ------------------

# name = "non_interpolated_vpinns"
# name = "vpinns"
name = "rvpinns"
# name = "vpinns_tanh"
save_dir = "figures"
os.makedirs(save_dir, exist_ok=True)

# ------------------ NN SOLUTION ------------------

# # Fractura 1
# fig = plt.figure(dpi=200)
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_trisurf(vertices_fracture_1.numpy(force=True)[:, 0],
#                 vertices_fracture_1.numpy(force=True)[:, 1],
#                 u_NN_fracture_1.reshape(-1).numpy(force=True),
#                 triangles=triangles_fracture_1.numpy(force=True),
#                 cmap='viridis', edgecolor='black', linewidth=0.1)
# # ax.set_title("Fracture 1")
# ax.set_xlabel(r"$x$")
# ax.set_ylabel(r"$y$")
# ax.set_zlabel(r"$u_h(x,y)$")
# ax.tick_params(labelsize=8)
# plt.tight_layout()
# plt.savefig(os.path.join(save_dir, f"{name}_nn_solution_fracture_1.png"))
# 

# # Fractura 2
# fig = plt.figure(dpi=200)
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_trisurf(vertices_fracture_2.numpy(force=True)[:, 0],
#                 vertices_fracture_2.numpy(force=True)[:, 1],
#                 u_NN_fracture_2.reshape(-1).numpy(force=True),
#                 triangles=triangles_fracture_2.numpy(force=True),
#                 cmap='viridis', edgecolor='black', linewidth=0.1)
# # ax.set_title("Fracture 2")
# ax.set_xlabel(r"$x$")
# ax.set_ylabel(r"$y$")
# ax.set_zlabel(r"$u_h(x,y)$")
# ax.tick_params(labelsize=8)
# plt.tight_layout()
# plt.savefig(os.path.join(save_dir, f"{name}_nn_solution_fracture_2.png"))
# 

vertices = torch.unbind(mesh.local_triangulations["vertices_3D"], dim = 0)

triangles =  torch.unbind(mesh.local_triangulations["triangles"], dim = 0)

solution = torch.unbind(u_NN_local, dim = 0)

plotter =pv.Plotter(off_screen=True)

for i in range(2):  # Para cada fractura
    verts = vertices[i].numpy(force=True)  # (N_v, 3)
    tris = triangles[i].numpy(force=True)  # (N_T, 3)
    sol = solution[i].numpy(force=True) 
    
    # pyvista espera una lista plana con un prefijo indicando el número de vértices por celda (3 para triángulos)
    faces = np.hstack([np.full((tris.shape[0], 1), 3), tris]).flatten()
    
    # Crear una malla PolyData
    mesh = pv.PolyData(verts, faces)
    
    mesh.point_data["solution"] = sol
    
    plotter.add_mesh(
    mesh,
    show_edges=True,
    scalars="solution",
    cmap="viridis",  # Puedes cambiar el colormap
    opacity=1,
    scalar_bar_args={"title": "Pressure"},
    lighting=False,
)

plotter.show()
plotter.screenshot(os.path.join(save_dir, f"{name}_solution.png"))

# ------------------ TRACES (JUMPS) ------------------

plt.figure(dpi=200)

plt.plot(trace_nodes, 
         u_NN_trace, 
         linestyle='--')

plt.xlabel("trace length")
plt.ylabel("value")
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, f"{name}_trace_value_nn.png"))

# ------------------ TRACES (JUMPS) ------------------

# Fractura 1
fig = plt.figure(dpi=200)
plt.plot(points_trace_fracture_1.numpy(force=True),
         jump_u_trace_fracture_1.reshape(-1).numpy(force=True),
         color="black", label=r"$u^{ex}$")
plt.scatter(points_trace_fracture_1.numpy(force=True),
            jump_u_NN_trace_fracture_1.reshape(-1).numpy(force=True),
            color="r", label=r"$u_h$")
plt.xlabel("trace length")
plt.ylabel("jump value")
# plt.title("Fracture 1")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, f"{name}_trace_jump_fracture_1.png"))

# Fractura 2
fig = plt.figure(dpi=200)
plt.plot(points_trace_fracture_2.numpy(force=True),
         jump_u_trace_fracture_2.reshape(-1).numpy(force=True),
         color="black", label=r"$u^{ex}$")
plt.scatter(points_trace_fracture_2.numpy(force=True),
            jump_u_NN_trace_fracture_2.reshape(-1).numpy(force=True),
            color="r", label=r"$u_h$")
plt.xlabel("trace length")
plt.ylabel("jump value")
# plt.title("Fracture 2")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, f"{name}_trace_jump_fracture_2.png"))


# ------------------ ERROR EVOLUTION ------------------

# Relative Loss & H1 Error (semilogy)
fig = plt.figure(dpi=300)
plt.semilogy(relative_loss_list, label=r"$\frac{\sqrt{\mathcal{L}(u_\theta)}}{\|u\|_{U}}$", linestyle="-.")
plt.semilogy(H1_error_list, label=r"$\frac{\|u-u_\theta\|_{U}}{\|u\|_{U}}$", linestyle=":")
plt.legend(fontsize=10)
plt.tight_layout()
plt.ylabel("Value")
plt.xlabel("# Epochs")
plt.savefig(os.path.join(save_dir, f"{name}_error_evolution.png"))


# # Relative Loss vs H1 Error (loglog)
# fig = plt.figure(dpi=300)
# plt.loglog(relative_loss_list, H1_error_list)
# plt.title(f"Error vs Loss comparison of VPINNs ({name})")
# plt.xlabel("Relative Loss")
# plt.ylabel("Relative Error")
# plt.tight_layout()
# plt.savefig(os.path.join(save_dir, f"{name}_error_vs_loss.png"))
# 

# ------------------ RELATIVE ERROR ------------------

# Convert to numpy
H1_error_fracture_1 = H1_error_fracture_1.numpy(force=True)
c4e_fracture_1 = c4e_fracture_1.numpy(force=True)
H1_error_fracture_2 = H1_error_fracture_2.numpy(force=True)
c4e_fracture_2 = c4e_fracture_2.numpy(force=True)

# Shared color scale
all_errors = np.concatenate([H1_error_fracture_1, H1_error_fracture_2])
norm = colors.Normalize(vmin=all_errors.min(), vmax=all_errors.max())
cmap = cm.viridis

# Fractura 1
fig, ax = plt.subplots(dpi=200)
face_colors = cmap(norm(H1_error_fracture_1))
collection = PolyCollection(c4e_fracture_1, facecolors=face_colors,
                            edgecolors='black', linewidths=0.2)
ax.add_collection(collection)
ax.autoscale()
ax.set_aspect('equal')
ax.set_xlim([-1, 1])
ax.set_ylim([0, 1])
# ax.set_title("Fracture 1")
ax.tick_params(labelsize=8)
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
fig.colorbar(sm, ax=ax, orientation='vertical', label=r'$H^1$ relative error', fraction=0.025)
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, f"{name}_relative_error_fracture_1.png"))


fig, ax = plt.subplots(dpi=200)
face_colors = cmap(norm(H1_error_fracture_2))
collection = PolyCollection(c4e_fracture_2, facecolors=face_colors,
                            edgecolors='black', linewidths=0.2)
ax.add_collection(collection)
ax.autoscale()
ax.set_aspect('equal')
ax.set_xlim([-1, 1])
ax.set_ylim([0, 1])
# ax.set_title("Fracture 2")
ax.tick_params(labelsize=8)
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
fig.colorbar(sm, ax=ax, orientation='vertical', label=r'$H^1$ relative error', fraction=0.025)
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, f"{name}_relative_error_fracture_2.png"))

