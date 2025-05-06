import torch

class Mesh:
    def __init__(self, 
                 coords4nodes: torch.Tensor, 
                 nodes4elements: torch.Tensor,
                 dirichlet: torch.Tensor):

        self.update_mesh(coords4nodes, 
                         nodes4elements,
                         dirichlet)
        
    def update_mesh(self, coords4nodes, nodes4elements, dirichlet):   
        
        self.coords4nodes = coords4nodes
        self.nodes4elements = nodes4elements
        
        self.coords4elements = self.coords4nodes[self.nodes4elements]
                
        self.nb_nodes, self.nb_dimensions = self.coords4nodes.shape
        self.nb_elements, self.nb_size4elements = self.nodes4elements.shape
        
        self.rows_idx = nodes4elements.repeat(1, self.nb_size4elements).reshape(-1)
        self.cols_idx = nodes4elements.repeat_interleave(self.nb_size4elements).reshape(-1)
        
        self.form_idx = self.nodes4elements.reshape(-1)
        
        self.dirichlet = dirichlet
        
        self.inner_dofs = torch.arange(self.nb_nodes)[~torch.isin(torch.arange(self.nb_nodes), self.dirichlet)]
            
class Elements:
    def __init__(self,
                 P_order: int,
                 int_order: int):
        
        self.P_order = P_order
        self.int_order = int_order
    
        self.compute_gauss_values(self.int_order)
                
    def shape_functions_value_and_grad(self, x, y):
        
        lambda_1 = 1.0 - x - y
        lambda_2 = x
        lambda_3 = y
        
        grad_lambda_1 = torch.tensor([-1.0, -1.0])
        grad_lambda_2 = torch.tensor([ 1.0,  0.0])
        grad_lambda_3 = torch.tensor([ 0.0,  1.0])

        if self.P_order == 1:
            phi = torch.stack([lambda_1, 
                               lambda_2, 
                               lambda_3], dim = -1)
            
            id_xy = torch.ones_like(torch.concat([x,y], dim = -1))
            
            grad_phi_1 = grad_lambda_1 * id_xy
            grad_phi_2 = grad_lambda_2 * id_xy
            grad_phi_3 = grad_lambda_3 * id_xy
            
            grad_phi = torch.stack([grad_phi_1,
                                     grad_phi_2,
                                     grad_phi_3], dim = -1)
        
        return phi, grad_phi
        
    def compute_gauss_values(self, int_order):
        
        if int_order == 1: 
            self.gaussian_nodes_x = torch.tensor([[1/3]])
                                                  
            self.gaussian_nodes_y = torch.tensor([[1/3]])
            
            self.gaussian_weights = torch.tensor([1.])
            
        if int_order == 2:
            self.gaussian_nodes_x = torch.tensor([[1/6], [2/3], [1/6]])
                                                  
            self.gaussian_nodes_y = torch.tensor([[1/6], [1/6], [2/3]])
            
            self.gaussian_weights = torch.tensor([1/3, 1/3, 1/3])
            
        if int_order == 3: 
            self.gaussian_nodes_x = torch.tensor([[1/3], [0.6], [0.2], [0.2]])
                                                  
            self.gaussian_nodes_y = torch.tensor([[1/3], [0.2], [0.6], [0.2]])
            
            self.gaussian_weights = torch.tensor([-9/16, 25/48, 25/48, 25/48])
            
    def compute_integral_values(self, mesh: Mesh):
        
        self.v, shape_functions_grad = self.shape_functions_value_and_grad(self.gaussian_nodes_x, self.gaussian_nodes_y)
                
        self.mapping_jacobian =  shape_functions_grad @ mesh.coords4elements.unsqueeze(-3) 
        
        self.det_map_jacobian = abs(torch.linalg.det(self.mapping_jacobian)).unsqueeze(-1)
        
        self.integration_points = torch.split((self.v @ mesh.coords4elements.unsqueeze(-3)), 1, dim = -1)
        
        self.inv_mapping_jacobian = torch.linalg.inv(self.mapping_jacobian)
        
        self.v_grad = shape_functions_grad.mT @ self.inv_mapping_jacobian
                    
class Basis:
    def __init__(self, 
                 mesh: Mesh,
                 elements: Elements):
        
        self.mesh = mesh
        self.elements = elements
        self.elements.compute_integral_values(self.mesh)
        
    def integrate_functional(self, function):
                
        integral_value = 0.5 * self.elements.gaussian_weights.unsqueeze(-1) * function(self.elements).sum(-2, keepdim = True) * self.elements.det_map_jacobian
                        
        return integral_value
        
    def integrate_lineal_form(self, function):
        
        integral_value = torch.zeros(self.mesh.nb_nodes, 1)
        
        integrand_value = (0.5 * self.elements.gaussian_weights * function(self.elements) * self.elements.det_map_jacobian).sum(1, keepdim = True)
        
        integral_value.index_put_((self.mesh.form_idx,), 
                              integrand_value.reshape(-1, 1),
                              accumulate = True)
                
        return integral_value
        
    def integrate_bilineal_form(self, function):
        
        global_matrix = torch.zeros(self.mesh.nb_nodes, self.mesh.nb_nodes)
        
        local_matrix = 0.5 * self.elements.gaussian_weights * function(self.elements) * self.elements.det_map_jacobian
        
        global_matrix.index_put_((self.mesh.rows_idx, self.mesh.cols_idx), 
                              local_matrix.reshape(-1),
                              accumulate = True)
                
        return global_matrix