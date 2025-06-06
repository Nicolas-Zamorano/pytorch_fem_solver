import torch
from abc import ABC, abstractmethod

class Abstract_Mesh(ABC):
    def __init__(self,
                 coords4nodes: torch.Tensor,
                 nodes4elements: torch.Tensor):
        
       self.coords4nodes = coords4nodes
       self.nodes4elements = nodes4elements.unsqueeze(-2)
       
       self.coords4elements = self.coords4nodes[self.nodes4elements]
       
       self.mesh_parameters = self.compute_mesh_parameters(self.coords4nodes, 
                                                           self.nodes4elements)
       
       self.elements_diameter, self.nodes4boundary, self.edges_parameters = self.compute_edges_values(self.coords4nodes, 
                                                                                                      self.nodes4elements, 
                                                                                                      self.mesh_parameters)
           
    def compute_edges_values(self, coords4nodes: torch.Tensor, nodes4elements: torch.Tensor, mesh_parameters: torch.Tensor):
    
        nodes4edges = nodes4elements[..., self.edges_permutations]
        
        coords4edges = coords4nodes[nodes4edges]
                        
        elements_diameter = torch.min(torch.norm(coords4edges[..., 1, :] - coords4edges[..., 0, :], dim = -1, keepdim = True), dim = -2)[0]       
        
        nodes4unique_edges, edges_idx, boundary_mask = torch.unique(nodes4edges.reshape(-1, mesh_parameters["nb_dimensions"]).mT, 
                                                                              return_inverse = True, 
                                                                              sorted = False, 
                                                                              return_counts = True, 
                                                                              dim = -1)
        
        nodes4unique_edges = nodes4unique_edges.mT
                
        nodes4boundary_edges = nodes4unique_edges[boundary_mask == 1]
        nodes4inner_edges = nodes4unique_edges[boundary_mask != 1]
        
        elements4boundary_edges = (nodes4boundary_edges.unsqueeze(-2).unsqueeze(-2) == nodes4elements.unsqueeze(-3).unsqueeze(-1)).any(dim = -2).all(dim = -1).float().argmax(dim = -1) 
        elements4inner_edges = torch.nonzero((nodes4inner_edges.unsqueeze(-2).unsqueeze(-2) == nodes4elements.unsqueeze(-3).unsqueeze(-1)).any(dim = -2).all(dim = -1))[:, 1].reshape(-1, mesh_parameters["nb_dimensions"])
        
        nodes4boundary = torch.unique(nodes4boundary_edges)
        nodes_idx4boundary_edges = torch.nonzero((nodes4unique_edges.unsqueeze(-2) == nodes4boundary_edges.unsqueeze(-3)).all(dim = -1).any(dim = -1))

        edges_parameters = {"nodes4edges": nodes4edges,
                            "edges_idx": edges_idx,
                            "nodes4unique_edges": nodes4unique_edges,
                            "elements4boundary_edges": elements4boundary_edges,
                            "elements4inner_edges": elements4inner_edges,
                            "nodes_idx4boundary_edges": nodes_idx4boundary_edges}
 
        return elements_diameter, nodes4boundary, edges_parameters
       
    @abstractmethod
    def compute_mesh_parameters(self, coords4nodes: torch.Tensor, nodes4elements: torch.Tensor):
        raise NotImplementedError

class Mesh_Tri(Abstract_Mesh):
    def __init__(self, 
                 coords4nodes: torch.Tensor, 
                 nodes4elements: torch.Tensor):

        self.edges_permutations = torch.tensor([[0, 1], 
                                                [1, 2], 
                                                [0, 2]])
        
        super().__init__(coords4nodes, 
                         nodes4elements)

    @staticmethod
    def compute_mesh_parameters(coords4nodes: torch.Tensor, nodes4elements: torch.Tensor):
        
        nb_nodes, nb_dimensions = coords4nodes.shape
        nb_simplex = nodes4elements.shape[-3]       
        
        mesh_parameters = {"nb_nodes": nb_nodes,
                           "nb_dimensions": nb_dimensions,
                           "nb_simplex": nb_simplex}
        
        return mesh_parameters

    def map_fine_mesh(self, fine_mesh: torch.Tensor):
        """
        Devuelve un tensor de shape (n_elements_fino,) tal que 
        cada entrada i indica a qué triángulo del mallado grueso pertenece 
        el triángulo i del mallado fino.
        """
        c4e_h = fine_mesh.coords4elements        # (n_elem_h, 3, 2)
        c4e_H = self.coords4elements      # (n_elem_H, 3, 2)
        centroids_h = c4e_h.mean(dim = -2)          # (n_elem_h, 2)
    
        # Expandimos para broadcasting
        P = centroids_h[:, None, :]              # (n_elem_h, 1, 2)
        A = c4e_H[None, :, 0, :]                 # (1, n_elem_H, 2)
        B = c4e_H[None, :, 1, :]
        C = c4e_H[None, :, 2, :]
    
        v0 = C - A                               # (1, n_elem_H, 2)
        v1 = B - A
        v2 = P - A                               # (n_elem_h, n_elem_H, 2)
    
        dot00 = (v0 * v0).sum(dim = -1)            # (1, n_elem_H)
        dot01 = (v0 * v1).sum(dim = -1)
        dot11 = (v1 * v1).sum(dim = -1)
    
        dot02 = (v0 * v2).sum(dim = -1)            # (n_elem_h, n_elem_H)
        dot12 = (v1 * v2).sum(dim = -1)
    
        denom = dot00 * dot11 - dot01 * dot01    # (1, n_elem_H)
        denom = denom.clamp(min=1e-14)
    
        u = (dot11 * dot02 - dot01 * dot12) / denom  # (n_elem_h, n_elem_H)
        v = (dot00 * dot12 - dot01 * dot02) / denom
    
        inside = (u >= 0) & (v >= 0) & (u + v <= 1)   # (n_elem_h, n_elem_H)
    
        # Inicializar con -1
        mapping = torch.full((c4e_h.shape[0],), -1, dtype = torch.long)
    
        # Para cada triángulo fino, buscamos el primer triángulo grueso que lo contiene
        candidates = inside.nonzero(as_tuple=False)  # shape (n_matches, 2)
    
        # candidates[i, 0] es índice en T_h, candidates[i, 1] es índice en T_H
        # Queremos quedarnos con el primer T_H válido para cada T_h
        seen = torch.zeros(c4e_h.shape[0], dtype = torch.bool)
        for i in range(candidates.shape[0]):
            idx_h, idx_H = candidates[i]
            if not seen[idx_h]:
                mapping[idx_h] = idx_H
                seen[idx_h] = True
    
        return mapping

class Elements_1D:
    def __init__(self,
                 P_order: int,
                 int_order: int):
        
        self.P_order = P_order
        self.int_order = int_order
        
        self.compute_barycentric_coordinates = lambda x: torch.concat([0.5 * (1. - x), 
                                                                       0.5 * (1. + x)], dim = -1)
        
        self.barycentric_grad = torch.tensor([[-0.5],
                                              [ 0.5]])
        
        self.compute_gauss_values(self.int_order)
        
    def compute_gauss_values(self, int_order: int):
        
        if int_order == 2: 
            
            nodes = 1./torch.sqrt(torch.tensor(3.))
            
            self.gaussian_nodes= torch.tensor([[-nodes], 
                                               [nodes]])
                                                              
            self.gaussian_weights = torch.tensor([[[.5]],
                                                  [[.5]]])
        
        if int_order == 3:
            
            nodes = torch.sqrt(torch.tensor(3/5))
            
            self.gaussian_nodes = torch.tensor([[0], 
                                                [-nodes], 
                                                [nodes]])
            
            self.gaussian_weights = torch.tensor([[[8/18]],
                                                  [[5/18]],
                                                  [[5/18]]])
            
    # def shape_functions_value_and_grad(self, bar_coords: torch.Tensor, inv_map_jacobian: torch.Tensor):
        
    #     if self.P_order == 1: 
            
    #         v = bar_coords.unsqueeze(0).repeat(inv_map_jacobian.shape[0], 1, 1)
            
    #         v_grad = (self.barycentric_grad @ inv_map_jacobian)
        
    #     return v, v_grad
    
    def compute_integral_values(self, coords4elements):
                
        self.bar_coords = self.compute_barycentric_coordinates(self.gaussian_nodes) 
                        
        self.map_jacobian = coords4elements.mT @ self.barycentric_grad
        
        self.det_map_jacobian = torch.linalg.norm(self.map_jacobian, dim = -2, keepdim = True)
                
        self.integration_points = torch.split((self.bar_coords @ coords4elements).unsqueeze(-1).unsqueeze(-4), 1, dim = -2)
        
        self.inv_map_jacobian = 1./self.det_map_jacobian
                
        # self.v, self.v_grad = self.shape_functions_value_and_grad(self.bar_coords, self.inv_map_jacobian)

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
            
            v = bar_coords.unsqueeze(-1)
            
            v_grad = self.barycentric_grad @ inv_map_jacobian
            
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
                                      4 * (lambda_1 * grad_lambda_3 + lambda_3 * grad_lambda_1)], dim = -2) @ inv_map_jacobian

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
            
    def compute_shape_functions(self, x, y, inv_map_jacobian):
        
        bar_coords = self.compute_barycentric_coordinates(x, y) 
        
        v, v_grad = self.shape_functions_value_and_grad(bar_coords, inv_map_jacobian)

        return bar_coords, v, v_grad                 
       
    def compute_map(self, coords4elements, nb_simplex):
        
        map_jacobian =  coords4elements.mT @ self.barycentric_grad
        
        det_map_jacobian = abs(torch.linalg.det(map_jacobian)).reshape(nb_simplex, 1, 1, 1)
        
        inv_map_jacobian = torch.linalg.inv(map_jacobian)
        
        return map_jacobian, det_map_jacobian, inv_map_jacobian
            
    def compute_integral_values(self, mesh: Mesh):
        
        self.map_jacobian, self.det_map_jacobian, self.inv_map_jacobian = self.compute_map(mesh.coords4elements, 
                                                                                           mesh.nb_simplex)
                
        self.bar_coords, self.v, self.v_grad = self.compute_shape_functions(self.gaussian_nodes_x, 
                                                                            self.gaussian_nodes_y, 
                                                                            self.inv_map_jacobian.unsqueeze(-3))
                        
        self.integration_points = torch.split((self.bar_coords @ mesh.coords4elements).unsqueeze(-1), 1, dim = -2)
                        
    def compute_inverse_map(self, first_node, integration_points = None, inv_map_jacobian = None):

        inv_map = inv_map_jacobian @ (integration_points - first_node) 
                
        return inv_map

class Interior_Facet_Basis:
    def __init__(self, 
                 mesh: Mesh,
                 elements: Elements_1D):
        
        self.elements = elements
        self.mesh = mesh
        
        self.compute_dofs(mesh)

        self.elements.compute_integral_values(mesh.coords4nodes[mesh.nodes4inner_edges])
        
    def compute_dofs(self, mesh: Mesh):
                        
        if self.elements.P_order == 1:
            
            self.update_dofs_values(mesh.coords4nodes, mesh.nodes4inner_edges, mesh.nodes4boundary)
            
    def update_dofs_values(self, coords4dofs, nodes4dofs, nodes4boundary_dofs):
        
        self.coords4global_dofs = coords4dofs
        self.global_dofs4elements = nodes4dofs
        self.nodes4boundary_dofs = nodes4boundary_dofs

        self.coords4elements = self.coords4global_dofs[self.global_dofs4elements]
                
        self.nb_global_dofs, self.nb_dimensions = self.coords4global_dofs.shape
        self.nb_elements, self.nb_local_dofs = self.global_dofs4elements.shape
        
        self.rows_idx = self.global_dofs4elements.repeat(1, self.nb_local_dofs, 1).reshape(-1)
        self.cols_idx = self.global_dofs4elements.repeat_interleave(self.nb_local_dofs, 1).reshape(-1)
        
        self.form_idx = self.global_dofs4elements.reshape(-1)
                
        self.inner_dofs = torch.arange(self.nb_global_dofs)[~torch.isin(torch.arange(self.nb_global_dofs), self.nodes4boundary_dofs)]
     
    def integrate_functional(self, function, *args, **kwargs):
                
        integral_value = (2. * self.elements.gaussian_weights.squeeze(-1) * function(self.elements, *args, **kwargs) * self.elements.det_map_jacobian.unsqueeze(-1)).sum(-2)
                        
        return integral_value    
    
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
            
            new_coords4dofs = (mesh.coords4nodes[mesh.nodes4unique_edges]).mean(-2)
            new_nodes4dofs = mesh.edges_idx.reshape(mesh.nb_simplex, 3) + mesh.nb_nodes
            new_boundary_dofs = mesh.nodes_idx4boundary_edges.squeeze(-1) + mesh.nb_nodes
            
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
                
    def interpolate(self, basis, tensor = None):
        
        if basis == self:
            dofs_idx = self.global_dofs4elements.unsqueeze(-3) 
            v = self.elements.v
            v_grad = self.elements.v_grad()
            
        else:
        
            if basis.__class__ == Basis:
                
                elements_mask = self.mesh.map_fine_mesh(basis.mesh)
                
                dofs_idx = self.global_dofs4elements[elements_mask].unsqueeze(-2)
    
            if basis.__class__ == Interior_Facet_Basis: 
    
                elements_mask = basis.mesh.elements4inner_edges
                
                dofs_idx = basis.mesh.nodes4elements[elements_mask].unsqueeze(-2)
                        
            coords4elements_first_node = self.coords4elements[:, [0], :][elements_mask].unsqueeze(-1)
    
            integration_points = torch.concat(basis.elements.integration_points, dim = -2)
    
            inv_map_jacobian = self.elements.inv_map_jacobian[elements_mask].unsqueeze(-3)
    
            
            new_integrations_points = self.elements.compute_inverse_map(coords4elements_first_node,
                                                                        integration_points, 
                                                                        inv_map_jacobian)
            
            _, v, v_grad = self.elements.compute_shape_functions(*torch.unbind(new_integrations_points,dim = -2), 
                                                                          inv_map_jacobian)
                    
        if tensor != None:
            
            interpolation = (tensor[dofs_idx] * v).sum(-2, keepdim = True)
            
            interpolation_grad = (tensor[dofs_idx] * v_grad).sum(-2, keepdim = True)
            
            return interpolation, interpolation_grad
        
        else:
            
            nodes = torch.split(self.coords4global_dofs, 1, dim = -1)
            
            interpolator = lambda function : (function(*nodes)[dofs_idx] * v).sum(-2, keepdim = True)
            
            interpolator_grad = lambda function : (function(*nodes)[dofs_idx] * v_grad).sum(-2, keepdim = True)
            
            return interpolator, interpolator_grad