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