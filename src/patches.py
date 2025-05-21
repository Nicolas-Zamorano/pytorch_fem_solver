import torch

class Patches:
    def __init__(self, 
                 centers: torch.Tensor, 
                 radius: torch.Tensor):
        
        self.centers = centers
        self.radius = radius
                
        self.signs4nodes = torch.tensor([[-1, -1],
                                         [ 1, -1],
                                         [ 1,  1],
                                         [-1,  1],
                                         [ 0,  0]], dtype = torch.int64)
        
        self.nodes4elements = torch.tensor([[0, 1, 4],
                                            [1, 2, 4],
                                            [2, 3, 4],
                                            [0, 3, 4]], dtype = torch.int64)
        
        coords4nodes = self.centers.unsqueeze(-2) + self.signs4nodes * self.radius.unsqueeze(-2)
                
        self.compute_values(coords4nodes)
        
    def compute_values(self, coords4nodes: torch.Tensor):

        self.coords4nodes = coords4nodes
        
        self.coords4elements = self.coords4nodes[:, self.nodes4elements]
                        
        self.nb_patches, self.nb_nodes, self.nb_dimensions = self.coords4nodes.shape
        self.nb_elements, self.size4elements = self.nodes4elements.shape

        self.nodes4edges = self.nodes4elements[..., 
                                               [[0, 1], 
                                                [1, 2], 
                                                [0, 2]]].reshape(-1, self.nb_dimensions).mT
        
        self.nodes4unique_edges, self.edges_idx, self.counts = torch.unique(self.nodes4edges, 
                                                                            return_inverse = True, 
                                                                            sorted = False, 
                                                                            return_counts = True, 
                                                                             dim = -1)
                
        self.nodes4boundary_edges = self.nodes4unique_edges[..., self.counts == 1]
        self.nodes4boundary = torch.unique(self.nodes4boundary_edges)
        
        self.boundary_edges_idx = torch.nonzero((self.nodes4unique_edges.unsqueeze(-1) == self.nodes4boundary_edges.unsqueeze(-2)).all(dim = -3).any(dim = -1)).squeeze(-1)

    def refine_patches(self, refine_idx):
        
        new_radius = 0.5 * self.radius[refine_idx]
        
        new_centers = self.centers[refine_idx, :].unsqueeze(-2) + self.signs4nodes[:-1, :] * new_radius.unsqueeze(-2) # WARNING!!!  no deja los centros en la ordenados correctamente, los ordenados en sentido horario (originalmente los etiuqetamos sentido antihorairo). 
        
        new_coords4nodes = new_centers.unsqueeze(-2) + (self.signs4nodes * new_radius.unsqueeze(-2)).unsqueeze(-3)
                
        refined_radius = torch.concat([self.radius[~refine_idx], new_radius.repeat(4, 1)], dim = 0)
        
        refined_centers = torch.concat([self.centers[~refine_idx, :], new_centers.view(-1, self.nb_dimensions)], dim = 0)
        
        refined_coords4nodes = torch.concat([self.coords4nodes[~refine_idx, ...], new_coords4nodes.view(-1, self.nb_nodes, self.nb_dimensions)], dim = 0)
                
        refined_radius = torch.concat([self.radius, new_radius.repeat(4, 1)], dim = 0)
        
        refined_centers = torch.concat([self.centers, new_centers.view(-1, self.nb_dimensions)], dim = 0)
        
        refined_coords4nodes = torch.concat([self.coords4nodes, new_coords4nodes.view(-1, self.nb_nodes, self.nb_dimensions)], dim = 0)

        return refined_centers, refined_radius, refined_coords4nodes
    
    def uniform_refine(self, nb_refinements: int = 1):
       
       for i in range(nb_refinements):
           self.centers, self.radius, new_coords4nodes = self.refine_patches(torch.tensor([True] * self.nb_patches))
           self.compute_values(new_coords4nodes)
           
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
        
    def shape_functions_value_and_grad(self, bar_coords: torch.Tensor, inv_mapping_jacobian: torch.Tensor):
        
        if self.P_order == 1: 
            
            v = bar_coords.unsqueeze(0).unsqueeze(-1).repeat(inv_mapping_jacobian.shape[0], inv_mapping_jacobian.shape[1], 1, 1, 1)
            
            v_grad = (self.barycentric_grad @ inv_mapping_jacobian).unsqueeze(2).repeat(1, 1, bar_coords.shape[0], 1, 1)
            
        return v, v_grad
    
    def compute_gauss_values(self, int_order: int):
        
        if int_order == 2:
            
            self.gaussian_nodes_x = torch.tensor([[1/6], [2/3], [1/6]])
                                                  
            self.gaussian_nodes_y = torch.tensor([[1/6], [1/6], [2/3]])
            
            self.gaussian_weights = torch.tensor([[[1/3]], [[1/3]], [[1/3]]])
            
    def compute_integral_values(self, patches: Patches):
                
        self.bar_coords = self.compute_barycentric_coordinates(self.gaussian_nodes_x, self.gaussian_nodes_y) 
                        
        self.mapping_jacobian =  patches.coords4elements.mT @ self.barycentric_grad
        
        self.det_map_jacobian = abs(torch.linalg.det(self.mapping_jacobian)).reshape(patches.nb_patches, patches.nb_elements, 1, 1, 1)
        
        self.integration_points = torch.split((self.bar_coords @ patches.coords4elements).unsqueeze(-1), 1, dim = -2)
        
        self.inv_mapping_jacobian = torch.linalg.inv(self.mapping_jacobian)
                
        self.v, self.v_grad = self.shape_functions_value_and_grad(self.bar_coords, self.inv_mapping_jacobian)
        
class Basis:
    def __init__(self, 
                 patches: Patches,
                 elements: Elements):
        
        self.elements = elements
        self.patches = patches
        
        self.compute_dofs(patches)

        self.elements.compute_integral_values(self.patches)
        
        self.patches_area = 0.5 * self.elements.det_map_jacobian.sum(-4).squeeze(-1)
        
    def update_dofs_values(self, coords4dofs, nodes4dofs, nodes4boundary_dofs):
        
        self.coords4global_dofs = coords4dofs
        self.global_dofs4elements = nodes4dofs
        self.nodes4boundary_dofs = nodes4boundary_dofs

        self.coords4elements = self.coords4global_dofs[:, self.global_dofs4elements]
                
        _, self.nb_global_dofs, self.nb_dimensions = self.coords4global_dofs.shape
        self.nb_elements, self.nb_local_dofs = self.global_dofs4elements.shape
        
        self.rows_idx = self.global_dofs4elements.repeat(1, self.nb_local_dofs).reshape(-1)
        self.cols_idx = self.global_dofs4elements.repeat_interleave(self.nb_local_dofs).reshape(-1)
        self.patches_idx = torch.arange(self.patches.nb_patches).unsqueeze(-1)

        
        self.form_idx = self.global_dofs4elements.reshape(-1)
                
        self.inner_dofs = torch.arange(self.nb_global_dofs)[~torch.isin(torch.arange(self.nb_global_dofs), self.nodes4boundary_dofs)]
    

    def compute_dofs(self, patches: Patches):
                
        if self.elements.P_order == 1:
            
            self.update_dofs_values(patches.coords4nodes, patches.nodes4elements, patches.nodes4boundary)
    
    def integrate_functional(self, function, *args, **kwargs):
                
        integral_value = (0.5 * self.elements.gaussian_weights * function(self.elements, *args, **kwargs) * self.elements.det_map_jacobian).sum(-3)
                        
        return integral_value
        
    def integrate_lineal_form(self, function,  *args, **kwargs):
        
        integral_value = torch.zeros(self.patches.nb_patches, self.nb_global_dofs, 1)
        
        integrand_value = (0.5 * self.elements.gaussian_weights * function(self.elements, *args, **kwargs) * self.elements.det_map_jacobian).sum(-3)
        
        integral_value.index_put_((self.patches_idx, self.form_idx,), 
                              integrand_value.reshape(self.patches.nb_patches, -1, 1),
                              accumulate = True)
                
        return integral_value
        
    def integrate_bilineal_form(self, function, *args, **kwargs):
        
        global_matrix = torch.zeros(self.patches.nb_patches, self.nb_global_dofs, self.nb_global_dofs)
        
        local_matrix = (0.5 * self.elements.gaussian_weights * function(self.elements, *args, **kwargs) * self.elements.det_map_jacobian).sum(-3)
        
        global_matrix.index_put_((self.patches_idx, self.rows_idx, self.cols_idx), 
                              local_matrix.reshape(self.patches.nb_patches,-1),
                              accumulate = True)
        
        return global_matrix