import torch
import math
import patches
import fem

import matplotlib.pyplot as plt

from Neural_Network import Neural_Network
from Triangulation import Triangulation

import skfem

from datetime import datetime

torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
torch.set_default_dtype(torch.float64)

#---------------------- Neural Network Functions ----------------------#

def NN_gradiant(NN, x, y):

    x.requires_grad_(True)
    y.requires_grad_(True)
    
    output = NN.forward(x, y)
    
    gradients = torch.autograd.grad(outputs = output,
                                    inputs = (x, y),
                                    grad_outputs = torch.ones_like(output),
                                    retain_graph = True,
                                    create_graph = True)
        
    return torch.concat(gradients, dim = -1)

def optimizer_step(optimizer, scheduler, loss_value):
        optimizer.zero_grad()
        loss_value.backward(retain_graph = True)
        optimizer.step()
        scheduler.step()

#---------------------- Neural Network Parameters ----------------------#

epochs = 1
learning_rate = 0.5e-3
decay_rate = 0.99
decay_steps = 100

input_dimension = 2
output_dimension = 1
deep_layers = 4
hidden_layers_dimension = 40

    #---------------------- VPINNs/RVPINNs ----------------------#

NN = torch.jit.script(Neural_Network(input_dimension = input_dimension, 
                                     output_dimension = output_dimension,
                                     deep_layers = deep_layers, 
                                     hidden_layers_dimension = hidden_layers_dimension))

optimizer = torch.optim.Adam(NN.parameters(), 
                             lr = learning_rate)  

scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 
                                                   decay_rate ** (1/decay_steps))

    #---------------------- MF-VPINNs ----------------------#

NN_MF_VPINNs = torch.jit.script(Neural_Network(input_dimension = input_dimension, 
                                               output_dimension = output_dimension,
                                               deep_layers = deep_layers, 
                                               hidden_layers_dimension = hidden_layers_dimension))

optimizer_MF_VPINNs = torch.optim.Adam(NN_MF_VPINNs.parameters(), 
                             lr = learning_rate)  

scheduler_MF_VPINNs = torch.optim.lr_scheduler.ExponentialLR(optimizer_MF_VPINNs , 
                                                             decay_rate ** (1/decay_steps))

    #---------------------- MF-RVPINNs ----------------------#  

NN_MF_RVPINNs = torch.jit.script(Neural_Network(input_dimension = input_dimension, 
                                                output_dimension = output_dimension,
                                                deep_layers = deep_layers, 
                                                hidden_layers_dimension = hidden_layers_dimension))

optimizer_MF_RVPINNs = torch.optim.Adam(NN_MF_RVPINNs.parameters(), 
                                 lr = learning_rate) 

scheduler_MF_RVPINNs = torch.optim.lr_scheduler.ExponentialLR(optimizer_MF_RVPINNs, 
                                                       decay_rate ** (1/decay_steps))

#---------------------- FEM Parameters ----------------------#

coords4nodes, nodes4elements = Triangulation(3)

mesh_sk = skfem.MeshTri1().refined(3)

# coords4nodes = mesh_sk.p.T

# nodes4elements = mesh_sk.t.T

mesh = fem.Mesh(torch.tensor(coords4nodes), torch.tensor(nodes4elements))

elements = fem.Elements(P_order = 1, 
                        int_order = 2)

V = fem.Basis(mesh, elements)

centers = torch.tensor([[0.5, 0.5]])
radius = torch.tensor([[0.5]])

mesh_MF = patches.Patches(centers, radius)

# mesh_MF.plot_patches()

mesh_MF.uniform_refine(2)

elements_MF = patches.Elements(P_order = 1, 
                               int_order = 2)

V_MF = patches.Basis(mesh_MF, elements_MF)

nodes_array = V.mesh.coords4nodes.cpu()
triangles_array = V.mesh.nodes4elements.cpu()
dofs_array = V.coords4global_dofs.cpu()

fig, ax = plt.subplots()
ax.triplot(nodes_array[:, 0], nodes_array[:, 1], triangles_array, color='black', lw=1)
ax.plot(dofs_array[:, 0], dofs_array[:, 1], 'o', color='red')
ax.set_aspect('equal')

plt.show()

#---------------------- Residual Parameters ----------------------#

rhs = lambda x, y: 5. * math.pi**2 * torch.sin(2 * math.pi * x) * torch.sin(math.pi * y)

def residual(elements, NN):
    
    x, y = elements.integration_points
    
    NN_grad = NN_gradiant(NN, x, y)
    
    v = elements.v
    v_grad = elements.v_grad
    rhs_value = rhs(x, y)
            
    return rhs_value * v - v_grad @ NN_grad.mT

def gram_matrix(elements):
    
    v = elements.v
    v_grad = elements.v_grad
    
    return v_grad @ v_grad.mT + v @ v.mT

A_MF = V_MF.integrate_bilineal_form(gram_matrix)[:, V_MF.inner_dofs, :][:, :, V_MF.inner_dofs]

A_inv_MF = torch.linalg.inv(A_MF)

A = V.integrate_bilineal_form(gram_matrix)[V.inner_dofs, :][:, V.inner_dofs]

A_inv = torch.linalg.inv(A)

#---------------------- Error Parameters ----------------------#

exact = lambda x, y : torch.sin(2 * math.pi * x) * torch.sin(math.pi * y)

exact_dx = lambda x, y : 2 * math.pi * torch.cos(2 * math.pi * x) * torch.sin(math.pi * y)
exact_dy = lambda x, y : math.pi * torch.sin(2 * math.pi * x) * torch.cos(math.pi * y)

def H1_exact(elements):
    
    x, y = elements.integration_points
    
    # exact = torch.sin(5 * math.pi * x) * torch.sin(math.pi * y)
    
    # exact_dx = 5 * math.pi * torch.cos(5 * math.pi * x) * torch.sin(math.pi * y)
    # exact_dy = math.pi * torch.sin(5 * math.pi * x) * torch.cos(math.pi * y)
    
    return exact_dx(x, y)**2 + exact_dy(x, y)**2 + exact(x, y)**2

def H1_norm(elements, NN):
   
    x, y = elements.integration_points
    
    NN_dx, NN_dy = torch.split(NN_gradiant(NN,x, y), 1 , dim = -1)

    # exact = torch.sin(math.pi * x) * torch.sin(math.pi * y)
    
    # exact_dx = 5 * math.pi * torch.cos(5 * math.pi * x) * torch.sin(math.pi * y)
    # exact_dy = math.pi * torch.sin(5 * math.pi * x) * torch.cos(math.pi * y)

    L2_error = (exact(x, y) - NN(x,y))**2
    
    H1_0_error = (exact_dx(x, y) - NN_dx)**2 + (exact_dy(x, y) - NN_dy)**2
    
    return L2_error + H1_0_error 
    
exact_norm = torch.sqrt(torch.sum(V.integrate_functional(H1_exact)))

#---------------------- Training ----------------------#

H1_error_list = []
H1_error_MF_RVPINNs_list = []
H1_error_MF_VPINNs_list = []

start_time = datetime.now()

for epoch in range(epochs):
    current_time = datetime.now().strftime("%H:%M:%S")
    print(f"{'='*20} [{current_time}] Epoch:{epoch + 1}/{epochs} {'='*20}")
    
    #---------------------- VPINNs/RVPINNs ----------------------#

    residual_value = V.integrate_lineal_form(residual, NN)[V.inner_dofs]
        
    loss_value = residual_value.T @ (A_inv @ residual_value)
    
    # loss_value = (residual_value**2).sum()

    optimizer_step(optimizer,scheduler, loss_value)
    
    error_norm = torch.sqrt(torch.sum(V.integrate_functional(H1_norm, NN)))/exact_norm
    
    H1_error_list.append(error_norm.item())
        
     #---------------------- MF-VPINNs ----------------------#

    residual_value_MF_VPINNs = V_MF.integrate_lineal_form(residual, NN_MF_VPINNs)[:, V_MF.inner_dofs]
            
    loss_value_MF_VPINNs  = (residual_value_MF_VPINNs **2/V_MF.patches_area).sum()
    
    optimizer_step(optimizer_MF_VPINNs, scheduler_MF_VPINNs, loss_value_MF_VPINNs)

    error_norm_MF_VPINNs = torch.sqrt(torch.sum(V.integrate_functional(H1_norm, NN_MF_VPINNs)))/exact_norm
    
    H1_error_MF_VPINNs_list.append(error_norm_MF_VPINNs.item())

    #---------------------- MF-RVPINNs ----------------------#        
    
    residual_value_MF_RVPINNs = V_MF.integrate_lineal_form(residual, NN_MF_RVPINNs)[:, V_MF.inner_dofs]
        
    loss_value_MF_RVPINNs = (residual_value_MF_RVPINNs.mT * (A_inv_MF * residual_value_MF_RVPINNs)).sum()
    
    optimizer_step(optimizer_MF_RVPINNs, scheduler_MF_RVPINNs, loss_value_MF_RVPINNs)
    
    error_norm_MF_RVPINNs = torch.sqrt(torch.sum(V.integrate_functional(H1_norm, NN_MF_RVPINNs)))/exact_norm
                            
    H1_error_MF_RVPINNs_list.append(error_norm_MF_RVPINNs.item())

    print(f"Mesh method error: {error_norm.item():.8f} VPINNs error: {error_norm_MF_VPINNs.item():.8f} RVPINNs error: {error_norm_MF_RVPINNs.item():.8f}")

end_time = datetime.now()

execution_time = end_time - start_time

print(f"Training time: {execution_time}")

#---------------------- Plotting ----------------------#

N_points = 100

x = torch.linspace(0, 1, N_points)
y = torch.linspace(0, 1, N_points)
X, Y = torch.meshgrid(x, y, indexing = "ij")

with torch.no_grad(): 
    Z = exact(X, Y)

figure_solution = plt.figure()
axis_solution = figure_solution.add_subplot(111, projection = '3d')

contour = axis_solution.plot_surface(X.cpu().detach().numpy(), 
                                     Y.cpu().detach().numpy(), 
                                     Z.cpu().detach().numpy(), 
                                     cmap = 'viridis')

axis_solution.set(title = "Exact solution",
                  xlabel = "x",
                  ylabel = "y",
                  zlabel = r"$u_x,y)$")

figure_error, axis_error = plt.subplots(dpi = 500)

axis_error.semilogy(H1_error_list,
                    # label = "VPINNs",
                    label = "RVPINNs",
                    linestyle = ":")

axis_error.semilogy(H1_error_MF_VPINNs_list,
                    label = "MF-VPINNs",
                    linestyle = "-.")

axis_error.semilogy(H1_error_MF_RVPINNs_list,
                    label = "MF-RVPINNs",
                    linestyle = "--")

axis_error.legend(fontsize=15)

axis_error.set(xlabel = "# Epochs",
               ylabel = "Relative Error")

plt.show()



