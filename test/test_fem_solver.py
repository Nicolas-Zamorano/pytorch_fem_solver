import torch
from math import pi

# torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
torch.set_default_dtype(torch.float64)

def shape_functions(x, y):
    
    psi_1 = 1 - x - y
    psi_2 = x
    psi_3 = y
    
    return torch.cat([psi_1, psi_2, psi_3], dim = 1)

def fem_solver(coords4nodes, nodes4elements, dirichlet, f):
    
    #--------------- Define Variables ---------#
    
    nb_nodes, nb_dimensions = coords4nodes.shape
    nb_elements, nb_size4elements = nodes4elements.shape
    
    rows_idx = nodes4elements.repeat(1, nb_size4elements).reshape(-1)
    cols_idx = nodes4elements.repeat_interleave(nb_size4elements).reshape(-1)
    form_idx = nodes4elements.reshape(-1)
    
    coords4elements = coords4nodes[nodes4elements]
    
    dof = torch.arange(nb_nodes)[~torch.isin(torch.arange(nb_nodes), dirichlet)]
    
    gaussian_nodes =  torch.tensor([[1/6, 1/6],
                                    [2/3, 1/6],
                                    [1/6, 2/3]])
    
    gaussian_weights = torch.tensor([[1/6], 
                                      [1/6], 
                                      [1/6]])
    
    # gaussian_nodes = torch.tensor([[1/3, 1/3],
    #                                 [0.6, 0.2],
    #                                 [0.2, 0.6],
    #                                 [0.2, 0.2]])
    
    # gaussian_weights = torch.tensor([[-9/32],
    #                                   [25/96],
    #                                   [25/96],
    #                                   [25/96]])
        
    # shape_functions_grad = torch.tensor([[-1., -1.],
    #                                      [ 1.,  0.],
    #                                      [ 0.,  1.]])
    
    #--------------- Compute integration values ---------#
    
    v = shape_functions(*torch.split(gaussian_nodes, 1, dim = -1))
        
    mapping_jacobian =  coords4elements.mT @ shape_functions_grad
    
    det_map_jacobian = abs(torch.linalg.det(mapping_jacobian)).reshape(nb_elements, 1, 1)
    
    integration_points = torch.split((v @ coords4elements), 1, dim = -1)
    
    inv_mapping_jacobian = torch.linalg.inv(mapping_jacobian)
    
    v_grad = (shape_functions_grad @ inv_mapping_jacobian)
    
    #--------------- Assembly matrix ---------# 
    
    stiff_matrix = torch.zeros(nb_nodes, nb_nodes)
    
    stiff_values = 0.5 * v_grad @ v_grad.mT * det_map_jacobian

    stiff_matrix.index_put_((rows_idx, cols_idx), 
                            stiff_values.reshape(-1), 
                            accumulate = True)
    
    #--------------- Assembly vector ----------#
    
    rhs = torch.zeros(nb_nodes, 1)
    
    integral_value = (gaussian_weights * f(*integration_points) @ v_grad.mT * det_map_jacobian).sum(1)
            
    rhs.index_put_((form_idx,), 
                   integral_value.reshape(-1, 1),
                   accumulate = True)
    
    #--------------- Solve system ---------------#

    solution = torch.zeros(nb_nodes, 1)
        
    solution[dof] = torch.linalg.solve(stiff_matrix[dof][:, dof], rhs[dof])
    
    return solution, stiff_matrix

coords4nodes = torch.tensor([[0., 0.],
        [1., 0.],
        [0., 1.],
        [1., 1.]])

nodes4elements = torch.tensor([[0, 1, 2],
                         [1, 2, 3]])

dirichlet = torch.tensor([0, 1, 2, 3])

def f(x,y):
    # return torch.ones_like(x)
    # return torch.concat([torch.ones_like(x),torch.ones_like(y)], dim = -1)
    return torch.concat([pi * torch.cos(pi * x)*torch.sin(pi * y), pi * torch.sin(pi * x) * torch.cos(pi * y)],dim = -1)

solution, A = fem_solver(coords4nodes, nodes4elements, dirichlet, f)

print(A)

import matplotlib.pyplot as plt

fig = plt.figure(figsize = (8, 6))
ax = fig.add_subplot(projection = '3d')

coords4nodes_numpy = coords4nodes.cpu().detach().numpy()
nodes4elements_numpy = nodes4elements.cpu().detach().numpy()
solution_numpy = solution.cpu().detach().numpy()

ax.plot_trisurf(coords4nodes_numpy[:, 0],
                coords4nodes_numpy[:, 1], 
                solution_numpy[:, 0], 
                triangles = nodes4elements_numpy, 
                cmap = 'viridis', 
                edgecolor = 'k')

plt.show()