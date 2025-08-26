import torch
import pickle

import matplotlib.pyplot as plt
import tensordict as td
import triangle as tr
import numpy as np
import os

from fracture_fem import Fractures, Element_Fracture, Fracture_Basis
from Neural_Network import Neural_Network_3D
from datetime import datetime

torch.set_default_dtype(torch.float64)
# torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
# torch.cuda.empty_cache()

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

epochs = 5000
learning_rate = 0.2e-3
decay_rate = 0.98
decay_steps = 200

NN = torch.jit.script(Neural_Network_3D(input_dimension = 3, 
                                        output_dimension = 1,
                                        deep_layers = 4, 
                                        hidden_layers_dimension = 15,
                                        activation_function= torch.nn.ReLU()))

#---------------------- FEM Parameters ----------------------#

h = 0.5

n = 3

fracture_2d_data = {"vertices" : [[-1., 0.],
                                  [ 1., 0.],
                                  [-1., 1.],
                                  [ 1., 1.],
                                  [ 0., 0.],
                                  [ 0., 1.]],
                    "segments" : [[0, 1],
                                  [1, 3],
                                  [2, 3],
                                  [0, 2],
                                  [4, 5]]
                    }

fractures_data = torch.tensor([[[-1., 0., 0.], 
                                [ 1., 0., 0.], 
                                [-1., 1., 0.], 
                                [ 1., 1., 0.]],
                               
                               [[ 0., 0.,-1.], 
                                [ 0., 0., 1.], 
                                [ 0., 1.,-1.], 
                                [ 0., 1., 1.]]])

fracture_triangulation = tr.triangulate(fracture_2d_data, 
                                                      "pqsea"+str(h**(n))
                                                      )

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

def a(basis):
    
    v_grad = basis.v_grad
    
    return v_grad @ v_grad.mT

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
    
NN_initial_parameters = NN.state_dict()

#---------------------- Solution ----------------------#

H1_norm_list = []
nb_dofs_list = []

for i in range(11):
    
    # torch.cuda.empty_cache()    
    NN.load_state_dict(NN_initial_parameters)
    
    optimizer = torch.optim.Adam(NN.parameters(), 
                                 lr = learning_rate)  

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 
                                                       decay_rate ** (1/decay_steps))

    fracture_triangulation = tr.triangulate(fracture_triangulation, 
                                                          "prsea"+str(h**(n+i))
                                                          )
    
    fracture_triangulation_torch = td.TensorDict((fracture_triangulation))
    
    fractures_triangulation = (fracture_triangulation_torch, fracture_triangulation_torch)
        
    mesh = Fractures(triangulations = fractures_triangulation,
                     fractures_3D_data = fractures_data)
    
    elements = Element_Fracture(P_order = 1, 
                                int_order = 2)
    
    V = Fracture_Basis(mesh, elements)
    
    exact_H1_norm = torch.sqrt(torch.sum(V.integrate_functional(H1_exact)))
    
    A = V.reduce(V.integrate_bilineal_form(a))

    A_inv = torch.linalg.inv(A)
    
    Ih, Ih_grad = V.interpolate(V)

    Ih_NN = lambda x, y, z : Ih(NN)
    Ih_grad_NN = lambda x, y, z : Ih_grad(NN)
    
    exact_norm = torch.sqrt(torch.sum(V.integrate_functional(H1_exact)))
    
    loss_opt = 10e4
    
    for epoch in range(epochs):
        current_time = datetime.now().strftime("%H:%M:%S")
        print(f"{'='*15} [{current_time}] Iter:{i} Epoch:{epoch + 1}/{epochs} {'='*15}")            
        residual_value = V.reduce(V.integrate_lineal_form(residual, Ih_grad_NN))
                                              
        # loss_value = residual_value.T @ (A_inv @ residual_value)
        
        loss_value = (residual_value**2).sum()
        
        print(f"Loss: {loss_value.item():.8f}")

        optimizer_step(optimizer, loss_value)
        
        if loss_value < loss_opt:
            loss_opt = loss_value
            params_opt = NN.state_dict()
        
    NN.load_state_dict(params_opt)
    
    H1_norm_value =  torch.sqrt(torch.sum(V.integrate_functional(H1_norm, Ih_NN, Ih_grad_NN)))/exact_norm
    
    H1_norm_list.append((H1_norm_value).item())
    
    nb_dofs_list.append(V.basis_parameters["nb_dofs"])
    
# ------------------ CONFIGURACIÓN ------------------

name = "vpinns"
# name = "rvpinns"
# name = "non_interpolated_vpinns"
save_dir = "figures"
os.makedirs(save_dir, exist_ok=True)

# ------------------ CONVERGENCIA VPINNs ------------------

nb_dofs_np = np.array(nb_dofs_list)
H1_norm_np = np.array(H1_norm_list)

log_dofs = np.log10(nb_dofs_np)
log_H1 = np.log10(H1_norm_np)
slope, intercept = np.polyfit(log_dofs, log_H1, 1)

H1_fit = 10**intercept * nb_dofs_np**slope

fig, ax = plt.subplots(dpi=200)
ax.loglog(nb_dofs_list,
          H1_norm_np,
          "^",
          color="orange",
          markersize=7,
          markeredgecolor="black",
          label=f"decay rate = {-slope:.2f}")

ax.loglog(nb_dofs_list,
          H1_fit,
          "-.",
          color="orange",
          alpha=0.5)

ax.set_xlabel("# DOFs")
ax.set_ylabel(r"$H^1$ Relative Error")
ax.legend()
ax.grid(True, which="both", linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, f"{name}_H1_convergence.png"))
plt.show()

# ------------------ GUARDAR DATOS ------------------

with open(os.path.join(save_dir, f"{name}_H1_norm_convergence.pkl"), "wb") as file:
    pickle.dump([nb_dofs_np, H1_norm_np], file)