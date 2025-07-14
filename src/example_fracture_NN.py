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

    output = NN.forward(x, y, z)
    
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

epochs = 10000
learning_rate = 0.2e-3
decay_rate = 0.99
decay_steps = 200

NN = torch.jit.script(Neural_Network_3D(input_dimension = 3, 
                                        output_dimension = 1,
                                        deep_layers = 4, 
                                        hidden_layers_dimension = 25,
                                        activation_function= torch.nn.ReLU()))

optimizer = torch.optim.Adam(NN.parameters(), 
                             lr = learning_rate)  

scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 
                                                   decay_rate ** (1/decay_steps))

#---------------------- FEM Parameters ----------------------#

h = 0.5

n = 9

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

# Ih_NN = lambda x, y, z : Ih(NN)
# Ih_grad_NN = lambda x, y, z : Ih_grad(NN)

Ih_NN = lambda x, y, z : NN(x, y, z)
Ih_grad_NN = lambda x, y, z : NN_gradiant(NN, x, y, z)

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

def H1_norm(basis, u_NN, u_NN_grad):
    
    exact_value = exact(*basis.integration_points)
    
    u_NN_value = u_NN(*basis.integration_points)
    
    L2_error = (exact_value - u_NN_value)**2

    exact_dx_value, exact_dy_value, exact_dz_value = torch.split(exact_grad(*basis.integration_points), 1, dim = -1)
    
    u_NN_dx, u_NN_dy, u_NN_dz = torch.split(u_NN_grad(*basis.integration_points), 1, dim = -1)
           
    H1_0_error = (exact_dx_value - u_NN_dx)**2 + (exact_dy_value - u_NN_dy)**2 + (exact_dz_value - u_NN_dz)**2
    
    return H1_0_error + L2_error
            
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

H1_error_fracture_1, H1_error_fracture_2 =  torch.unbind(torch.sqrt(V.integrate_functional(H1_norm, Ih_NN, Ih_grad_NN)/V.integrate_functional(H1_exact)), dim = 0)

c4e_fracture_1, c4e_fracture_2 =  torch.unbind(mesh.local_triangulations["coords4triangles"], dim = 0)

#---------------------- Plot ----------------------#

### --- PLOT FEM SOLUTION --- ###

fig_sol = plt.figure(figsize = (15, 4), dpi = 200)
fig_sol.suptitle("NN solution", fontsize = 16)

ax_fracture_1 = fig_sol.add_subplot(1, 2, 1, projection = '3d')
ax_fracture_1.plot_trisurf(vertices_fracture_1.numpy(force = True)[:, 0], 
                 vertices_fracture_1.numpy(force = True)[:, 1], 
                 u_NN_fracture_1.reshape(-1).numpy(force = True), 
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
                 u_NN_fracture_2.reshape(-1).numpy(force = True),
                 triangles = triangles_fracture_2.numpy(force = True),
                 cmap = 'viridis', 
                 edgecolor = 'black', 
                 linewidth = 0.1)

ax_fracture_2.set_title("Fracture 2")
ax_fracture_2.set_xlabel(r"$x$")
ax_fracture_2.set_ylabel(r"$y$")
ax_fracture_2.set_zlabel(r"$u_h(x,y)$")

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
                           jump_u_NN_trace_fracture_1.reshape(-1).numpy(force = True),
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
                           jump_u_NN_trace_fracture_2.reshape(-1).numpy(force = True),
                           color = "r",
                           label = r"$u_h$")
ax_jump_fracture_2.set_title("Fracture 2")
ax_jump_fracture_2.set_xlabel("trace lenght")
ax_jump_fracture_2.set_ylabel("jump value")
ax_jump_fracture_2.legend()

plt.show()

### --- PLOT TRAINING --- ###

figure_error, axis_error = plt.subplots(dpi = 500)

axis_error.semilogy(relative_loss_list,
                    label = r"$\frac{\sqrt{\mathcal{L}(u_\theta)}}{\|u\|_{U}}$",
                    linestyle = "-.")

axis_error.semilogy(H1_error_list,
                    label = r"$\frac{\|u-u_\theta\|_{U}}{\|u\|_{U}}$",
                    linestyle = ":")

axis_error.legend(fontsize=15)

figure_loglog, axis_loglog = plt.subplots()

axis_loglog.loglog(relative_loss_list,
                    H1_error_list)

axis_loglog.set(title = "Error vs Loss comparasion of RVPINNs method",
                xlabel = "Relative Loss", 
                ylabel = "Relative Error")

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

fig_error.suptitle("Relative error for NN solution", fontsize = 14)

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
                          label = 'Relative error')

plt.show()