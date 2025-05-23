import torch

class Mesh:
    def __init__(self, 
                 coords4nodes: torch.Tensor, 
                 nodes4elements: torch.Tensor):

        self.edges_permutations = torch.tensor([[0, 1], 
                                                [1, 2], 
                                                [0, 2]])

        self.compute_values(coords4nodes, nodes4elements)
        
    def compute_mesh_parameters(self):
        
        self.nb_nodes, self.nb_dimensions = self.coords4nodes.shape
        self.nb_simplex, self.size4simplex = self.nodes4elements.shape
        
    def compute_edges_values(self):
        
        self.nodes4edges = self.nodes4elements[..., self.edges_permutations]
        
        self.coords4edges = self.coords4nodes[self.nodes4edges]
                        
        self.elements_diameter = torch.max(torch.norm(self.coords4edges[..., 1, :] - self.coords4edges[..., 0, :], dim = -1, keepdim = True), dim = -2)[0]       
        
        nodes4unique_edges, self.edges_idx, self.boundary_mask = torch.unique(self.nodes4edges.reshape(-1, self.nb_dimensions).mT, 
                                                                              return_inverse = True, 
                                                                              sorted = False, 
                                                                              return_counts = True, 
                                                                              dim = -1)
        
        self.nodes4unique_edges = nodes4unique_edges.mT
                
        self.nodes4boundary_edges = self.nodes4unique_edges[self.boundary_mask == 1]
        self.nodes4inner_edges = self.nodes4unique_edges[self.boundary_mask != 1]
        
        self.coords4unique_edges = self.coords4nodes[self.nodes4unique_edges]
        
        self.nodes4boundary = torch.unique(self.nodes4boundary_edges)
        
    def compute_normals(self):
        
        # Compute unit normal vector from all edges.
                
        edge_vectors = self.coords4unique_edges[..., 1, :] - self.coords4unique_edges[..., 0, :]
        
        self.edges_length = torch.norm(edge_vectors, dim = -1, keepdim = True)
        
        self.boundary_edges_lenght = self.edges_length [self.boundary_mask == 1]
        self.inner_edges_lenght = self.edges_length [self.boundary_mask != 1]
        
        normal_vector = edge_vectors[..., [1, 0]] * torch.tensor([-1., 1.])

        unit_normal_vector = normal_vector / torch.norm(normal_vector, dim = -1, keepdim = True)
                
        self.normal4boundary_edges = unit_normal_vector[self.boundary_mask == 1]
        self.normal4inner_edges = unit_normal_vector[self.boundary_mask != 1]
        
        # Compute idx of interest.
        
        self.nodes_idx4boundary_edges = torch.nonzero((self.nodes4unique_edges.unsqueeze(-2) == self.nodes4boundary_edges.unsqueeze(-3)).all(dim = -1).any(dim = -2))
        
        self.elements4inner_edges = torch.nonzero((self.nodes4inner_edges.unsqueeze(-2).unsqueeze(-2) == self.nodes4elements.unsqueeze(-3).unsqueeze(-1)).any(dim = -2).all(dim = -1))[:, 1].reshape(-1, self.nb_dimensions)
        
        self.elements4boundary_edges = (self.nodes4boundary_edges.unsqueeze(-2).unsqueeze(-2) == self.nodes4elements.unsqueeze(-3).unsqueeze(-1)).any(dim = -2).all(dim = -1).float().argmax(dim = -1) 
    
        # Fix normal from boundary edges to point outside the domain.
    
        boundary_elements_centroid = self.coords4elements[self.elements4boundary_edges].mean(dim = -2)
        boundary_edges_midpoint = self.coords4nodes[self.nodes4boundary_edges].mean(dim = -2)
                
        boundary_direction_mask = (self.nodes4boundary_edges * (boundary_elements_centroid - boundary_edges_midpoint)).sum(dim = -1)
        
        self.normal4boundary_edges[boundary_direction_mask < 0] *= -1
        
        # Fix normal from inner edges to point to the other triangle.
    
        inner_elements_centroid = self.coords4elements[self.elements4inner_edges].mean(dim = -2)
                
        inner_direction_mask = (self.normal4inner_edges * (inner_elements_centroid[..., 1, :] - inner_elements_centroid[..., 0, :])).sum(dim = -1)
    
        self.normal4inner_edges[inner_direction_mask < 0] *= -1
    
    def compute_values(self, coords4nodes, nodes4elements):
        
        self.coords4nodes = coords4nodes
        self.nodes4elements = nodes4elements
        
        self.coords4elements = self.coords4nodes[self.nodes4elements]
                                
        self.compute_mesh_parameters()
        
        self.compute_edges_values()

        self.compute_normals()
        
class Elements:
    def __init__(self,
                 P_order: int,
                 int_order: int):
        
        self.P_order = P_order
        self.int_order = int_order
        
        self.compute_barycentric_coordinates = lambda x, y : torch.concat([1.0 - x - y, 
                                                                           x, 
                                                                           y], dim = -1)
        
        self.barycentric_grad = torch.tensor([[-1.0, -1.0],
                                              [ 1.0,  0.0],
                                              [ 0.0,  1.0]])
        
        self.compute_gauss_values(self.int_order)
                
    def shape_functions_value_and_grad(self, bar_coords: torch.Tensor, inv_map_jacobian: torch.Tensor):
        
        if self.P_order == 1: 
            
            v = bar_coords.unsqueeze(-1).repeat(inv_map_jacobian.shape[0], 1, 1, 1)
            
            v_grad = (self.barycentric_grad @ inv_map_jacobian).unsqueeze(1).repeat(1, bar_coords.shape[0], 1, 1)
            
        else:                   
        
            lambda_1, lambda_2, lambda_3 = torch.split(bar_coords, 1, dim = -1)
            
            grad_lambda_1, grad_lambda_2, grad_lambda_3 = torch.split(self.barycentric_grad, 1, dim = 0)
                    
            if self.P_order == 2:
                
                v = torch.stack([lambda_1 * (2 * lambda_1 - 1),
                                 lambda_2 * (2 * lambda_2 - 1),
                                 lambda_3 * (2 * lambda_3 - 1),
                                 4 * lambda_1 * lambda_2,
                                 4 * lambda_2 * lambda_3,
                                 4 * lambda_3 * lambda_1], dim = -2)
                
                v_grad = torch.stack([(4 * lambda_1 - 1) * grad_lambda_1,
                                      (4 * lambda_2 - 1) * grad_lambda_2,
                                      (4 * lambda_3 - 1) * grad_lambda_3,
                                      4 * (lambda_2 * grad_lambda_1 + lambda_1 * grad_lambda_2),
                                      4 * (lambda_3 * grad_lambda_2 + lambda_2 * grad_lambda_3),
                                      4 * (lambda_1 * grad_lambda_3 + lambda_3 * grad_lambda_1)], dim = -2) @ inv_map_jacobian.unsqueeze(1)

        return v, v_grad
                
    def compute_gauss_values(self, int_order: int):
        
        if int_order == 1: 
            
            self.gaussian_nodes_x = torch.tensor([[1/3]])
                                                  
            self.gaussian_nodes_y = torch.tensor([[1/3]])
            
            self.gaussian_weights = torch.tensor([[[1.]]])
            
        if int_order == 2:
            
            self.gaussian_nodes_x = torch.tensor([[1/6], [2/3], [1/6]])
                                                  
            self.gaussian_nodes_y = torch.tensor([[1/6], [1/6], [2/3]])
            
            self.gaussian_weights = torch.tensor([[[1/3]], [[1/3]], [[1/3]]])
            
        if int_order == 3: 
            
            self.gaussian_nodes_x = torch.tensor([[1/3], [0.6], [0.2], [0.2]])
                                                  
            self.gaussian_nodes_y = torch.tensor([[1/3], [0.2], [0.6], [0.2]])
            
            self.gaussian_weights = torch.tensor([[[-9/16]],
                                                  [[25/48]],
                                                  [[25/48]],
                                                  [[25/48]]])
                        
    def compute_integral_values(self, mesh: Mesh):
                
        self.bar_coords = self.compute_barycentric_coordinates(self.gaussian_nodes_x, self.gaussian_nodes_y) 
                        
        self.map_jacobian =  mesh.coords4elements.mT @ self.barycentric_grad
        
        self.det_map_jacobian = abs(torch.linalg.det(self.map_jacobian)).reshape(mesh.nb_simplex, 1, 1, 1)
        
        self.integration_points = torch.split((self.bar_coords @ mesh.coords4elements).unsqueeze(-1), 1, dim = -2)
        
        self.inv_map_jacobian = torch.linalg.inv(self.map_jacobian)
                
        self.v, self.v_grad = self.shape_functions_value_and_grad(self.bar_coords, self.inv_map_jacobian)

class Basis:
    def __init__(self, 
                 mesh: Mesh,
                 elements: Elements):
        
        self.elements = elements
        self.mesh = mesh
        
        self.compute_dofs(mesh)

        self.elements.compute_integral_values(self.mesh)
    
    def compute_new_dofs(self, mesh):
        
        if self.elements.P_order == 2:
            
            new_coords4dofs = (mesh.coords4nodes[mesh.nodes4unique_edges]).mean(0)
            new_nodes4dofs = mesh.edges_idx.reshape(mesh.nb_simplex, 3) + mesh.nb_nodes
            new_boundary_dofs = mesh.boundary_edges_idx + mesh.nb_nodes
            
        return new_coords4dofs, new_nodes4dofs, new_boundary_dofs
        
    def update_dofs_values(self, coords4dofs, nodes4dofs, nodes4boundary_dofs):
        
        self.coords4global_dofs = coords4dofs
        self.global_dofs4elements = nodes4dofs
        self.nodes4boundary_dofs = nodes4boundary_dofs

        self.coords4elements = self.coords4global_dofs[self.global_dofs4elements]
                
        self.nb_global_dofs, self.nb_dimensions = self.coords4global_dofs.shape
        self.nb_elements, self.nb_local_dofs = self.global_dofs4elements.shape
        
        self.rows_idx = self.global_dofs4elements.repeat(1, self.nb_local_dofs).reshape(-1)
        self.cols_idx = self.global_dofs4elements.repeat_interleave(self.nb_local_dofs).reshape(-1)
        
        self.form_idx = self.global_dofs4elements.reshape(-1)
                
        self.inner_dofs = torch.arange(self.nb_global_dofs)[~torch.isin(torch.arange(self.nb_global_dofs), self.nodes4boundary_dofs)]
    
    def compute_dofs(self, mesh: Mesh):
                
        if self.elements.P_order == 1:
            
            self.update_dofs_values(mesh.coords4nodes, mesh.nodes4elements, mesh.nodes4boundary)
            
        else:
            
            new_coords4dofs, new_nodes4dofs, new_nodes4boundary_dofs = self.compute_new_dofs(mesh)
            
            coords4dofs = torch.cat([mesh.coords4nodes, new_coords4dofs], dim = -2)
            nodes4dofs = torch.cat([mesh.nodes4elements, new_nodes4dofs], dim = -1)
            nodes4boundary_dofs = torch.cat([mesh.nodes4boundary, new_nodes4boundary_dofs], dim = -1)
            
            self.update_dofs_values(coords4dofs, nodes4dofs, nodes4boundary_dofs)

    def integrate_functional(self, function, *args, **kwargs):
                
        integral_value = (0.5 * self.elements.gaussian_weights * function(self.elements, *args, **kwargs) * self.elements.det_map_jacobian).sum(-3)
                        
        return integral_value
        
    def integrate_lineal_form(self, function,  *args, **kwargs):
        
        integral_value = torch.zeros(self.nb_global_dofs, 1)
        
        integrand_value = (0.5 * self.elements.gaussian_weights * function(self.elements, *args, **kwargs) * self.elements.det_map_jacobian).sum(-3)
        
        integral_value.index_put_((self.form_idx,), 
                              integrand_value.reshape(-1, 1),
                              accumulate = True)
                
        return integral_value
        
    def integrate_bilineal_form(self, function, *args, **kwargs):
        
        global_matrix = torch.zeros(self.nb_global_dofs, self.nb_global_dofs)
        
        local_matrix = (0.5 * self.elements.gaussian_weights * function(self.elements, *args, **kwargs) * self.elements.det_map_jacobian).sum(-3)
        
        global_matrix.index_put_((self.rows_idx, self.cols_idx), 
                              local_matrix.reshape(-1),
                              accumulate = True)
        
        return global_matrix
            
    def interpolate_to(self, elements):
        
        nodes_x, nodes_y = torch.split(self.coords4elements.unsqueeze(-3), 1, dim = -1)
        
        v, v_grad = self.elements.shape_functions_value_and_grad(elements.bar_coords, elements.inv_map_jacobian)
        
        interpolator = lambda function: (function(nodes_x, nodes_y) * v).sum(-2, keepdim = True)
        
        interpolator_grad = lambda function: (function(nodes_x, nodes_y) * v_grad).sum(-2, keepdim = True)
        
        return interpolator, interpolator_grad
    