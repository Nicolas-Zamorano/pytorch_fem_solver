import torch

import matplotlib.pyplot as plt
import tensordict as td
import triangle as tr

from Neural_Network import Neural_Network_3D
from fracture_fem import Fractures, Element_Fracture, Fracture_Basis
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

epochs = 1000
learning_rate = 0.5e-2
decay_rate = 0.98
decay_steps = 100

NN = torch.jit.script(Neural_Network_3D(input_dimension = 3, 
                                        output_dimension = 1,
                                        deep_layers = 4, 
                                        hidden_layers_dimension = 25))

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
    
    Ih_NN, Ih_grad_NN = V.interpolate(NN, is_array=False)
    
    NN_func = lambda x, y, z : Ih_NN
    NN_grad_func = lambda x, y, z : Ih_grad_NN
    
    NN_func = lambda x, y, z: NN(x, y, z)
    NN_grad_func = lambda x, y, z: NN_gradiant(NN, x, y, z)

    residual_value = V.reduce(V.integrate_lineal_form(residual, NN_grad_func))
                                          
    # loss_value = residual_value.T @ (A_inv @ residual_value)
    
    loss_value = (residual_value**2).sum()

    optimizer_step(optimizer, loss_value)
    
    error_norm = torch.sqrt(torch.sum(V.integrate_functional(H1_norm, NN_func, NN_grad_func)))/exact_norm
        
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

### --- FEM SOLUTION PARAMETERS --- ###

local_vertices_3D = V.global_triangulation["vertices_3D"][V.global_triangulation["global2local_idx"].reshape(2, -1)] 

vertices_fracture_1, vertices_fracture_2 = torch.unbind(mesh.local_triangulations["vertices"], dim = 0)

triangles_fracture_1, triangles_fracture_2 =  torch.unbind(mesh.local_triangulations["triangles"], dim = 0)

u_NN_local = NN(*torch.split(local_vertices_3D, 1, -1))

u_NN_fracture_1 , u_NN_fracture_2 = torch.unbind(u_NN_local, dim = 0)

### --- TRACE PARAMETERS --- ###

trace_nodes = V.global_triangulation["vertices_3D"][V.global_triangulation["traces_vertices"], 1].numpy(force = True)

exact_value_local = exact(*torch.split(local_vertices_3D, 1, -1))

exact_value_global = exact_value_local.reshape(-1,1)[V.global_triangulation["local2global_idx"]]

exact_trace = exact_value_global[V.global_triangulation["traces_vertices"]].numpy(force = True)

u_NN_global = u_NN_local.reshape(-1,1)[V.global_triangulation["local2global_idx"]]

u_NN_trace = u_NN_global[V.global_triangulation["traces_vertices"]].numpy(force = True)

### --- ERROR PARAMETERS --- ###

H1_error_fracture_1, H1_error_fracture_2 =  torch.unbind(torch.sqrt(V.integrate_functional(H1_norm, NN_func, NN_grad_func)/V.integrate_functional(H1_exact)), dim = 0)

c4e_fracture_1, c4e_fracture_2 =  torch.unbind(mesh.local_triangulations["coords4triangles"], dim = 0)

#---------------------- Plot ----------------------#

### --- PLOT FEM SOLUTION --- ###

fig_sol = plt.figure(figsize = (15, 4), dpi = 200)
fig_sol.suptitle("NN solution", fontsize = 16)

ax_fracture_1 = fig_sol.add_subplot(1, 3, 1, projection = '3d')
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

ax_fracture_2 = fig_sol.add_subplot(1, 3, 2, projection = '3d')
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

ax_trace = fig_sol.add_subplot(1, 3, 3)
ax_trace.plot(trace_nodes, 
              exact_trace, 
              label = r"$u$", 
              color='blue')

ax_trace.plot(trace_nodes, 
              u_NN_trace, 
              label = r"$u_h$", 
              color = 'red', 
              linestyle = '--')
ax_trace.set_title("value along the trace")
ax_trace.set_xlabel(r"$y$")
ax_trace.set_ylabel(r"u(x,y)")
ax_trace.legend()
ax_trace.grid(True)

plt.subplots_adjust(wspace = 0.4)  # Aumenta separación horizontal
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