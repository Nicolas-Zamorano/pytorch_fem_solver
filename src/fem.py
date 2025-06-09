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
        c4e_h = fine_mesh.coords4elements        # (n_elem_h, 1, 3, 2)
        c4e_H = self.coords4elements             # (n_elem_H, 1, 3, 2)
        centroids_h = c4e_h.mean(dim = -2).squeeze(1)  # (n_elem_h, 2)
    
        # Expandimos para broadcasting
        P = centroids_h[:, None, :]              # (n_elem_h, 1, 2)
        A = c4e_H[:, 0, 0, :][None, :, :]         # (1, n_elem_H, 2)
        B = c4e_H[:, 0, 1, :][None, :, :]
        C = c4e_H[:, 0, 2, :][None, :, :]
    
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


class Abstract_Element(ABC):
    def __init__(self,
                 P_order: int,
                 int_order: int):
        
        self.P_order = P_order
        self.int_order = int_order
        
        self.gaussian_nodes, self.gaussian_weights = self.compute_gauss_values()
        
    def compute_integral_values(self, coords4elements: torch.Tensor):
        
        det_map_jacobian, inv_map_jacobian = self.compute_map(coords4elements)
                
        bar_coords, v, v_grad = self.compute_shape_functions(self.gaussian_nodes, inv_map_jacobian)
                        
        integration_points = torch.split(bar_coords.mT @ coords4elements, 1, dim = -1)
                        
        dx = self.reference_element_area * self.gaussian_weights * det_map_jacobian 
        
        return v, v_grad, integration_points, dx
    
    def compute_map(self, coords4elements: torch.Tensor):
        
        map_jacobian = coords4elements.mT @ self.barycentric_grad
        
        det_map_jacobian, inv_map_jacobian = self.compute_det_and_inv_map(map_jacobian)
        
        return det_map_jacobian, inv_map_jacobian
            
    def compute_shape_functions(self, gaussian_nodes: torch.Tensor, inv_map_jacobian: torch.Tensor):
        
        bar_coords = self.compute_barycentric_coordinates(gaussian_nodes) 
        
        v, v_grad = self.shape_functions_value_and_grad(bar_coords, inv_map_jacobian)

        return bar_coords, v, v_grad                 

    @staticmethod
    def compute_inverse_map(first_node: torch.Tensor, integration_points: torch.Tensor, inv_map_jacobian: torch.Tensor):

        inv_map = inv_map_jacobian @ (integration_points - first_node) 
                
        return inv_map

    @abstractmethod
    def compute_gauss_values(self):
        raise NotImplementedError
        
    @abstractmethod
    def shape_functions_value_and_grad(self, bar_coords, inv_map_jacobian):
        raise NotImplementedError
        
    @abstractmethod
    def compute_det_and_inv_map(self, map_jacobian):
        raise NotImplementedError

class Element_Line(Abstract_Element):
    def __init__(self,
                 P_order: int = 1,
                 int_order: int = 2):
        
        self.compute_barycentric_coordinates = lambda x: torch.concat([0.5 * (1. - x), 
                                                                       0.5 * (1. + x)], dim = -1)
        
        self.barycentric_grad = torch.tensor([[-0.5],
                                              [ 0.5]])
        
        self.reference_element_area = 2.
        
        super().__init__(P_order, 
                         int_order)
        
    def compute_gauss_values(self):
        
        if self.int_order == 2: 
            
            nodes = 1./torch.sqrt(torch.tensor(3.))
            
            gaussian_nodes= torch.tensor([[-nodes], 
                                          [nodes]])
                                                              
            gaussian_weights = torch.tensor([[[.5]],
                                             [[.5]]])
        
        if self.int_order == 3:
            
            nodes = torch.sqrt(torch.tensor(3/5))
            
            gaussian_nodes = torch.tensor([[0], 
                                           [-nodes], 
                                           [nodes]])
            
            gaussian_weights = torch.tensor([[[8/18]],
                                             [[5/18]],
                                             [[5/18]]])
            
        return gaussian_nodes, gaussian_weights
                        
    
    def shape_functions_value_and_grad(self, bar_coords: torch.Tensor, inv_map_jacobian: torch.Tensor):
        
        if self.P_order == 1: 
            
            v = bar_coords
            
            v_grad = self.barycentric_grad @ inv_map_jacobian
        
        return v, v_grad
    
    def compute_det_and_inv_map(map_jacobian):

        raise NotImplementedError 
    
class Element_Tri(Abstract_Element):
    def __init__(self,
                 P_order: int,
                 int_order: int):
        
        self.compute_barycentric_coordinates = lambda x: torch.stack([1.0 - x[..., [0]] - x[..., [1]], 
                                                                       x[..., [0]], 
                                                                       x[..., [1]]], dim = -2)
        
        self.barycentric_grad = torch.tensor([[-1.0, -1.0],
                                              [ 1.0,  0.0],
                                              [ 0.0,  1.0]])
        
        self.reference_element_area = 0.5
        
        super().__init__(P_order, 
                         int_order)
                    
    def shape_functions_value_and_grad(self, bar_coords: torch.Tensor, inv_map_jacobian: torch.Tensor):
        
        if self.P_order == 1: 
            
            v = bar_coords
            
            v_grad = self.barycentric_grad @ inv_map_jacobian
            
        if self.P_order == 2:                   
        
            lambda_1, lambda_2, lambda_3 = torch.split(bar_coords, 1, dim = -2)
            
            grad_lambda_1, grad_lambda_2, grad_lambda_3 = torch.split(self.barycentric_grad, 1, dim = -2)
                    
            if self.P_order == 2:
                
                v = torch.concat([lambda_1 * (2 * lambda_1 - 1),
                                 lambda_2 * (2 * lambda_2 - 1),
                                 lambda_3 * (2 * lambda_3 - 1),
                                 4 * lambda_1 * lambda_2,
                                 4 * lambda_2 * lambda_3,
                                 4 * lambda_3 * lambda_1], dim = -2)
                
                v_grad = torch.concat([(4 * lambda_1 - 1) * grad_lambda_1,
                                      (4 * lambda_2 - 1) * grad_lambda_2,
                                      (4 * lambda_3 - 1) * grad_lambda_3,
                                      4 * (lambda_2 * grad_lambda_1 + lambda_1 * grad_lambda_2),
                                      4 * (lambda_3 * grad_lambda_2 + lambda_2 * grad_lambda_3),
                                      4 * (lambda_1 * grad_lambda_3 + lambda_3 * grad_lambda_1)], dim = -2) @ inv_map_jacobian

        return v, v_grad
                
    def compute_gauss_values(self):
        
        if self.int_order == 1: 
            
            gaussian_nodes = torch.tensor([[1/3, 1/3]])
                                                              
            gaussian_weights = torch.tensor([[[1.]]])
            
        if self.int_order == 2:
            
            gaussian_nodes = torch.tensor([[1/6, 1/6], 
                                           [2/3, 1/6], 
                                           [1/6, 2/3]])
                                                              
            gaussian_weights = torch.tensor([[[1/3]], 
                                             [[1/3]], 
                                             [[1/3]]])
            
        if self.int_order == 3: 
            
            gaussian_nodes = torch.tensor([[1/3, 1/3], 
                                           [0.6, 0.2], 
                                           [0.2, 0.6],
                                           [0.2, 0.2]])
                                                  
            
            gaussian_weights = torch.tensor([[[-9/16]],
                                             [[25/48]],
                                             [[25/48]],
                                             [[25/48]]])
            
        if self.int_order == 4:
            
            gaussian_nodes = torch.tensor([[0.816847572980459,  0.091576213509771],
                                           [0.091576213509771,  0.816847572980459],
                                           [0.091576213509771,  0.091576213509771],
                                           [0.108103018168070,  0.445948490915965],
                                           [0.445948490915965,  0.108103018168070],
                                           [0.445948490915965,  0.445948490915965]])
            
            gaussian_weights = torch.tensor([[[0.109951743655322]],
                                             [[0.109951743655322]],
                                             [[0.109951743655322]],
                                             [[0.223381589678011]],
                                             [[0.223381589678011]],
                                             [[0.223381589678011]]])
            
        return gaussian_nodes, gaussian_weights

    def compute_det_and_inv_map(self, map_jacobian: torch.Tensor):
        a = map_jacobian[..., [[0]], 0]
        b = map_jacobian[..., [[0]], 1]
        c = map_jacobian[..., [[1]], 0]
        d = map_jacobian[..., [[1]], 1]
    
        det_map_jacobian = a * d - b * c
    
        inv_map_jacobian = (1 / det_map_jacobian) * torch.concat([torch.concat([d, -b], dim = -1),
                                                                  torch.concat([-c, a], dim = -1)], dim = -2)
    
        return abs(det_map_jacobian), inv_map_jacobian

class Abstract_Basis(ABC):
    def __init__(self,
                 mesh: Abstract_Mesh,
                 elements: Abstract_Element):
        
        self.elements = elements
        self.mesh = mesh
            
        self.v, self.v_grad, self.integration_points, self.dx = elements.compute_integral_values(mesh.coords4elements)
        
        self.coords4global_dofs, self.global_dofs4elements, self.nodes4boundary_dofs = self.compute_dofs(mesh.coords4nodes, 
                                                                                                         mesh.nodes4elements, 
                                                                                                         mesh.nodes4boundary,
                                                                                                         mesh.mesh_parameters,
                                                                                                         mesh.edges_parameters,
                                                                                                         elements.P_order,)

        self.coords4elements = self.coords4global_dofs[self.global_dofs4elements]

        self.basis_parameters = self.compute_basis_parameters(self.coords4global_dofs, 
                                                              self.global_dofs4elements, 
                                                              self.nodes4boundary_dofs)

    def integrate_functional(self, function, *args, **kwargs):
                
        integral_value = (function(self, *args, **kwargs) * self.dx).sum(-3).sum(-2)
                        
        return integral_value
            
    def integrate_bilineal_form(self, function, *args, **kwargs):
        
        global_matrix = torch.zeros(self.basis_parameters["bilinear_form_shape"])
        
        local_matrix = (function(self, *args, **kwargs) * self.dx).sum(-3)
        
        global_matrix.index_put_(self.basis_parameters["bilinear_form_idx"], 
                              local_matrix.reshape(-1),
                              accumulate = True)
        
        return global_matrix
    
    def integrate_lineal_form(self, function,  *args, **kwargs):
        
        integral_value = torch.zeros(self.basis_parameters["linear_form_shape"])
        
        integrand_value = (function(self, *args, **kwargs) * self.dx).sum(-3)
        
        integral_value.index_put_(self.basis_parameters["linear_form_idx"], 
                              integrand_value.reshape(-1, 1),
                              accumulate = True)
                
        return integral_value
    
    @abstractmethod
    def compute_dofs(self, coords4nodes, nodes4elements, nodes4boundary, mesh_parameters, edges_parameters, P_order):
        raise NotImplementedError
        
    @abstractmethod
    def compute_basis_parameters(self, coords4global_dofs, global_dofs4elements, nodes4boundary_dofs):
        raise NotImplementedError

class Basis(Abstract_Basis):
    def __init__(self, 
                 mesh: Abstract_Mesh,
                 elements: Abstract_Element):
        
        super().__init__(mesh, 
                         elements)
    
    def compute_dofs(self, coords4nodes, nodes4elements, nodes4boundary, mesh_parameters, edges_parameters, P_order):
        
        if P_order == 1:
            
            coords4global_dofs = coords4nodes
            global_dofs4elements = nodes4elements
            nodes4boundary_dofs = nodes4boundary
        
        if P_order == 2:
            
            new_coords4dofs = (coords4nodes[edges_parameters["nodes4unique_edges"]]).mean(-2)
            new_nodes4dofs = edges_parameters["edges_idx"].reshape(mesh_parameters["nb_simplex"], 1, 3) + mesh_parameters["nb_nodes"]
            new_nodes4boundary_dofs = edges_parameters["nodes_idx4boundary_edges"].squeeze(-1) + mesh_parameters["nb_nodes"]
            
            coords4global_dofs = torch.cat([coords4nodes, new_coords4dofs], dim = -2)
            global_dofs4elements = torch.cat([nodes4elements, new_nodes4dofs], dim = -1)
            nodes4boundary_dofs = torch.cat([nodes4boundary, new_nodes4boundary_dofs], dim = -1)
            
        return coords4global_dofs, global_dofs4elements, nodes4boundary_dofs
            
    def compute_basis_parameters(self, coords4global_dofs, global_dofs4elements, nodes4boundary_dofs):
        
        nb_global_dofs = coords4global_dofs.shape[-2]
        nb_local_dofs = global_dofs4elements.shape[-1]
        
        inner_dofs = torch.arange(nb_global_dofs)[~torch.isin(torch.arange(nb_global_dofs), nodes4boundary_dofs)]

        rows_idx = global_dofs4elements.repeat(1, 1, nb_local_dofs).reshape(-1)
        cols_idx = global_dofs4elements.repeat_interleave(nb_local_dofs).reshape(-1)
        
        form_idx = global_dofs4elements.reshape(-1)
        
        basis_parameters = {"bilinear_form_shape" : (nb_global_dofs, nb_global_dofs),
                            "bilinear_form_idx": (rows_idx, cols_idx),
                            "linear_form_shape": (nb_global_dofs, 1),
                            "linear_form_idx": (form_idx,),
                            "inner_dofs": (inner_dofs)}

        return basis_parameters    

    def reduce(self, tensor):
        idx = self.basis_parameters["inner_dofs"]
        if tensor.shape[-1] != 1:
            return tensor[idx, :][:, idx]
        else:
            return tensor[idx]
                    
    def interpolate(self, basis, tensor = None):
        
        if basis == self:
            dofs_idx = self.global_dofs4elements
            v = self.v
            v_grad = self.v_grad
            
        else:
        
            if basis.__class__ == Basis:
                
                elements_mask = self.mesh.map_fine_mesh(basis.mesh)
                
                dofs_idx = self.global_dofs4elements[elements_mask].unsqueeze(-2)
    
            if basis.__class__ == Interior_Facet_Basis: 
    
                elements_mask = basis.mesh.elements4inner_edges
                
                dofs_idx = basis.mesh.nodes4elements[elements_mask].unsqueeze(-2)
                        
            coords4elements_first_node = self.coords4elements[:, [0], :][elements_mask]
        
            inv_map_jacobian = self.elements.inv_map_jacobian[elements_mask]
    
            
            new_integrations_points = self.elements.compute_inverse_map(coords4elements_first_node,
                                                                        basis.integration_points, 
                                                                        inv_map_jacobian)
            
            _, v, v_grad = self.elements.compute_shape_functions(new_integrations_points, inv_map_jacobian)
                    
        if tensor != None:
            
            interpolation = (tensor[dofs_idx] * v).sum(-2, keepdim = True)
            
            interpolation_grad = (tensor[dofs_idx] * v_grad).sum(-2, keepdim = True)
            
            return interpolation, interpolation_grad
        
        else:
            
            nodes = torch.split(self.coords4global_dofs, 1, dim = -1)
            
            interpolator = lambda function : (function(*nodes)[dofs_idx] * v).sum(-2, keepdim = True)
            
            interpolator_grad = lambda function : (function(*nodes)[dofs_idx] * v_grad).sum(-2, keepdim = True)
            
            return interpolator, interpolator_grad
        
class Interior_Facet_Basis(Abstract_Basis):
    def __init__(self, 
                 mesh: Abstract_Mesh,
                 elements: Element_Line):

        super().__init__(mesh, 
                         elements)

    def compute_basis_parameters(self, coords4global_dofs, global_dofs4elements, nodes4boundary_dofs):
        
        nb_global_dofs = coords4global_dofs.shape[-2]
        nb_local_dofs = global_dofs4elements.shape[-1]
        
        inner_dofs = torch.arange(nb_global_dofs)[~torch.isin(torch.arange(nb_global_dofs), nodes4boundary_dofs)]

        rows_idx = global_dofs4elements.repeat(1, nb_local_dofs).reshape(-1)
        cols_idx = global_dofs4elements.repeat_interleave(nb_local_dofs).reshape(-1)
        
        form_idx = global_dofs4elements.reshape(-1)

        basis_parameters = {"bilinear_form_shape" : (nb_global_dofs, nb_global_dofs),
                            "bilinear_form_idx": (rows_idx, cols_idx),
                            "linear_form_shape": (nb_global_dofs, 1),
                            "linear_form_idx": (form_idx),
                            "inner_dofs": inner_dofs}

        return basis_parameters       