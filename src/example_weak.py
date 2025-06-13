import torch
import math

import matplotlib.pyplot as plt
import triangle as tr

from Neural_Network import Neural_Network
from fem import Mesh_Tri, Element_Tri, Basis
from datetime import datetime

# torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
# torch.cuda.empty_cache()
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

def optimizer_step(optimizer, loss_value):
        optimizer.zero_grad()
        loss_value.backward(retain_graph = True)
        optimizer.step()
        scheduler.step()

#---------------------- Neural Network Parameters ----------------------#

epochs = 8000
learning_rate = 0.1e-2
decay_rate = 0.99
decay_steps = 100

NN = torch.jit.script(Neural_Network(input_dimension = 2, 
                                      output_dimension = 1,
                                      deep_layers = 4, 
                                      hidden_layers_dimension = 25))

optimizer = torch.optim.Adam(NN.parameters(), 
                             lr = learning_rate)  

scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 
                                                   decay_rate ** (1/decay_steps))

#---------------------- FEM Parameters ----------------------#

h = 0.5**1

mesh_data = tr.triangulate({"vertices":[[0., 0.],
                                        [1., 0.],
                                        [0., 1.],
                                        [1., 1.]]}, 
                            "Dqea"+str(h))

mesh = Mesh_Tri(triangulation = mesh_data)

elements = Element_Tri(P_order = 2, 
                       int_order = 4)

V = Basis(mesh, elements)

fig, ax_mesh = plt.subplots()

tr.plot(ax_mesh,**mesh_data)

#---------------------- Residual Parameters ----------------------#

rhs = lambda x, y : 2. * math.pi**2 * torch.sin(math.pi * x) * torch.sin(math.pi * y)

def residual(basis):
    
    x, y = basis.integration_points
    
    NN_grad = NN_gradiant(NN, x, y)
        
    v = basis.v
    v_grad = basis.v_grad
    rhs_value = rhs(x, y)
    
    return rhs_value * v - v_grad @ NN_grad.mT

def gram_matrix(basis):
    
    v = basis.v
    v_grad = basis.v_grad
    
    return v_grad @ v_grad.mT + v @ v.mT

A = V.reduce(V.integrate_bilineal_form(gram_matrix))

A_inv = torch.linalg.inv(A)

#---------------------- Error Parameters ----------------------#

exact = lambda x, y : torch.sin(math.pi * x) * torch.sin(math.pi * y)
exact_dx = lambda x, y : math.pi * torch.cos(math.pi * x) * torch.sin(math.pi * y)
exact_dy = lambda x, y : math.pi * torch.sin(math.pi * x) * torch.cos(math.pi * y)

def H1_exact(basis):

    x, y = basis.integration_points
    
    return exact(x, y)**2 + exact_dx(x, y)**2 + exact_dy(x,y)**2

def H1_norm(basis):
   
    x, y = basis.integration_points
    
    NN_dx, NN_dy = torch.split(NN_gradiant(NN, x, y), 1 , dim = -1)
    
    return (exact(x, y)- NN(x,y))**2  + (exact_dx(x, y) - NN_dx)**2 + (exact_dy(x, y) - NN_dy)**2
        
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

    residual_value = V.reduce(V.integrate_lineal_form(residual))
        
    loss_value = residual_value.T @ (A_inv @ residual_value)
    
    # loss_value = (residual_value**2).sum()

    optimizer_step(optimizer, loss_value)
    
    error_norm = torch.sqrt(torch.sum(V.integrate_functional(H1_norm)))/exact_norm
        
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

#---------------------- Plotting ----------------------#

NN.load_state_dict(params_opt)

N_points = 100

X, Y = torch.meshgrid(torch.linspace(0, 1, N_points), torch.linspace(0, 1, N_points), indexing = "ij")


with torch.no_grad(): 
    Z = abs(exact(X,Y) - NN(X, Y))

figure_solution, axis_solution = plt.subplots()

fig, ax = plt.subplots(dpi = 500)
c = ax.contourf(X.cpu(), Y.cpu(), Z.cpu(), levels = 100, cmap = 'viridis')
fig.colorbar(c, ax=ax, orientation='vertical')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title(r'$|u-u_\theta|$')
plt.tight_layout()

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