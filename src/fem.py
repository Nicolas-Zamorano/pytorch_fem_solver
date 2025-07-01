import torch
import tensordict
from abc import ABC, abstractmethod

torch.set_default_dtype(torch.float64)

class Abstract_Mesh(ABC):
    def __init__(self,
                 triangulation: dict):
        
        self.coords4nodes = triangulation["vertices"]
        self.nodes4elements = triangulation["triangles"]

        self.coords4elements = self.coords4nodes[self.nodes4elements]
        
        self.mesh_parameters = self.compute_mesh_parameters(self.coords4nodes, 
                                                           self.nodes4elements)
       
        self.elements_diameter, self.nodes4boundary, self.edges_parameters = self.compute_edges_values(self.coords4nodes, 
                                                                                                       self.nodes4elements, 
                                                                                                       self.mesh_parameters,
                                                                                                       triangulation)
               
    def compute_edges_values(self, coords4nodes: torch.Tensor, nodes4elements: torch.Tensor, mesh_parameters: torch.Tensor, triangulation: dict):
    
                        
        nodes4edges, _ = torch.sort(nodes4elements[..., self.edges_permutations], dim = -1)
        
        coords4edges = coords4nodes[nodes4edges]
        
        coords4edges_1, coords4edges_2 = torch.split(coords4edges, 1, dim = -2)
        
        elements_diameter = torch.min(torch.norm(coords4edges_2 - coords4edges_1, dim = -1, keepdim = True), dim = -2, keepdim = True)[0]       
        
        nodes4unique_edges = triangulation["edges"]
        boundary_mask = triangulation["edge_markers"].squeeze(-1)
        
        edges_idx = self.get_edges_idx(nodes4elements, nodes4unique_edges)

        nodes4boundary_edges = nodes4unique_edges[boundary_mask == 1]
        nodes4inner_edges = nodes4unique_edges[boundary_mask != 1]
        nodes4boundary = torch.nonzero(triangulation["vertex_markers"])[:, [0]]
                    
        elements4boundary_edges = (nodes4boundary_edges.unsqueeze(-2).unsqueeze(-2) == nodes4elements.unsqueeze(-1).unsqueeze(-4)).any(dim = -2).all(dim = -1).float().argmax(dim = -1,keepdim = True) 
        elements4inner_edges = torch.nonzero((nodes4inner_edges.unsqueeze(-2).unsqueeze(-2) == nodes4elements.unsqueeze(-1).unsqueeze(-4)).any(dim = -2).all(dim = -1), as_tuple=True)[1].reshape(-1, mesh_parameters["nb_dimensions"])
        
        nodes_idx4boundary_edges = torch.nonzero((nodes4unique_edges.unsqueeze(-2) == nodes4boundary_edges.unsqueeze(-3)).all(dim = -1).any(dim = -1))

        # compute inner edges normal vector
        
        coords4inner_edges = coords4nodes[nodes4inner_edges]
        
        coords4inner_edges_1, coords4inner_edges_2 = torch.split(coords4inner_edges, 1, dim = -2)

        inner_edges_vector = coords4inner_edges_2 - coords4inner_edges_1
        
        inner_edges_length = torch.norm(inner_edges_vector, dim = -1, keepdim = True)
                
        normal4inner_edges = inner_edges_vector[..., [1, 0]] * torch.tensor([-1., 1.])/ inner_edges_length
                
        inner_elements_centroid = self.coords4elements[elements4inner_edges].mean(dim = -2)
        
        inner_elements_centroid_1, inner_elements_centroid_2 = torch.split(inner_elements_centroid, 1 , dim = -2)
        
        inner_direction_mask = (normal4inner_edges * (inner_elements_centroid_2 - inner_elements_centroid_1)).sum(dim = -1)

        normal4inner_edges[inner_direction_mask < 0] *= -1

        edges_parameters = {"nodes4edges": nodes4edges,
                            "edges_idx": edges_idx,
                            "nodes4unique_edges": nodes4unique_edges,
                            "elements4boundary_edges": elements4boundary_edges,
                            "nodes4inner_edges": nodes4inner_edges,
                            "elements4inner_edges": elements4inner_edges,
                            "nodes_idx4boundary_edges": nodes_idx4boundary_edges,
                            "inner_edges_length": inner_edges_length,
                            "normal4inner_edges": normal4inner_edges}
 
        return elements_diameter, nodes4boundary, edges_parameters
    
    @abstractmethod
    def compute_mesh_parameters(self, coords4nodes: torch.Tensor, nodes4elements: torch.Tensor):
        raise NotImplementedError

class Mesh_Tri(Abstract_Mesh):
    def __init__(self, 
                 triangulation: dict):

        self.edges_permutations = torch.tensor([[0, 1], 
                                                [1, 2], 
                                                [0, 2]])
        
        super().__init__(triangulation)

    @staticmethod
    def compute_mesh_parameters(coords4nodes: torch.Tensor, nodes4elements: torch.Tensor):
        
        nb_nodes, nb_dimensions = coords4nodes.shape
        nb_simplex = nodes4elements.shape[-2]       
        
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
        c4e_H = self.coords4elements             # (n_elem_H, 3, 2)
        centroids_h = c4e_h.mean(dim = -2)       # (n_elem_h, 2)
    
        # Expandimos para broadcasting
        P = centroids_h[:, None, :]              # (n_elem_h, 1, 2)
        A = c4e_H[:, 0, 0, :][None, :, :]        # (1, n_elem_H, 2)
        B = c4e_H[:, 0, 1, :][None, :, :]
        C = c4e_H[:, 0, 2, :][None, :, :]
    
        v0 = C - A                               # (1, n_elem_H, 2)
        v1 = B - A
        v2 = P - A                               # (n_elem_h, n_elem_H, 2)
    
        dot00 = (v0 * v0).sum(dim = -1)          # (1, n_elem_H)
        dot01 = (v0 * v1).sum(dim = -1)
        dot11 = (v1 * v1).sum(dim = -1)
    
        dot02 = (v0 * v2).sum(dim = -1)          # (n_elem_h, n_elem_H)
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
    
        seen = torch.zeros(c4e_h.shape[0], dtype = torch.bool)
        for i in range(candidates.shape[0]):
            idx_h, idx_H = candidates[i]
            if not seen[idx_h]:
                mapping[idx_h] = idx_H
                seen[idx_h] = True
    
        return mapping
    
    def get_edges_idx(self, nodes4elements, nodes4unique_edges):
            # 1. Obtener los 3 edges de cada triángulo
            i0 = nodes4elements[..., 0]
            i1 = nodes4elements[..., 1]
            i2 = nodes4elements[..., 2]
        
            # Cada edge como par ordenado (min, max)
            tri_edges = torch.stack([
                torch.stack([torch.min(i0, i1), torch.max(i0, i1)], dim=1),
                torch.stack([torch.min(i1, i2), torch.max(i1, i2)], dim=1),
                torch.stack([torch.min(i2, i0), torch.max(i2, i0)], dim=1),
            ], dim=1)  # (n_triangles, 3, 2)
        
            # 2. Convertimos cada par (a,b) en una clave única: a * M + b
            M = nodes4elements.max().item() + 1  # M debe ser mayor al número de nodos
            tri_keys = tri_edges[:, :, 0] * M + tri_edges[:, :, 1]  # (n_triangles, 3)
        
            # 3. Hacer lo mismo con edges únicos
            edge_keys = (nodes4unique_edges.min(dim=-1).values * M + nodes4unique_edges.max(dim=-1).values) # (n_unique_edges,)
            
            # 4. Crear tabla de búsqueda
            sorted_keys, sorted_idx = torch.sort(edge_keys)  # Necesario para searchsorted
            flat_tri_keys = tri_keys.flatten()  # (n_triangles * 3,)
            
            # 5. Buscar cada key en sorted_keys
            edge_pos = torch.searchsorted(sorted_keys, flat_tri_keys)
            edge_indices = sorted_idx[edge_pos].reshape(tri_keys.shape)  # (n_triangles, 3)
        
            return edge_indices

class Fractures(Abstract_Mesh):
    def __init__(self, 
                 triangulations: tuple,
                 fractures_data: torch.Tensor):
        
        self.triangulations = triangulations
        self.fractures_data = fractures_data
        
        self.edges_permutations = torch.tensor([[0, 1], 
                                                [1, 2], 
                                                [0, 2]])
        
        self.triangulation =  self.stack_triangulations(triangulations)
        
        self.coords4nodes = self.triangulation["vertices"]
        self.nodes4elements = self.triangulation["triangles"]

        self.coords4elements = self.coords4nodes[torch.arange(self.coords4nodes.shape[0])[:, None, None],self.nodes4elements]
        
        self.mesh_parameters = self.compute_mesh_parameters(self.coords4nodes, 
                                                           self.nodes4elements)
       
        self.nodes4boundary = self.triangulation["nodes4boundary"]
        self.edges_parameters = dict()

    def compute_mesh_parameters(self, coords4nodes: torch.Tensor, nodes4elements: torch.Tensor):
                
        nb_fractures, nb_nodes, nb_dimensions = coords4nodes.shape
        _, nb_simplex, nb_size4simplex = nodes4elements.shape
        
        
        mesh_parameters = {"nb_fractures": nb_fractures,
                           "nb_nodes": nb_nodes,
                           "nb_dimensions": nb_dimensions,
                           "nb_simplex": nb_simplex,
                           "nb_size4simplex": nb_size4simplex}
        
        return mesh_parameters

    def stack_triangulations(self, fracture_triangulations: tensordict.TensorDict):
        
        global_vertices = torch.stack([triangulation["vertices"] for triangulation in fracture_triangulations], dim = 0)
        global_vertex_markers = torch.stack([triangulation["vertex_markers"] for triangulation in fracture_triangulations], dim = 0)
        
        global_triangles = torch.stack([triangulation["triangles"] for triangulation in fracture_triangulations], dim = 0)
        
        global_edges = torch.stack([triangulation["vertices"] for triangulation in fracture_triangulations], dim = 0)
        global_edge_markers = torch.stack([triangulation["edge_markers"] for triangulation in fracture_triangulations], dim = 0)

        global_nodes4boundary =  torch.stack([torch.nonzero(triangulation["vertex_markers"] == 1)[:, [0]] for triangulation in fracture_triangulations], dim = 0)

        global_triangulation = tensordict.TensorDict(
            dict(
                vertices = global_vertices,
                vertex_markers = global_vertex_markers,
                triangles = global_triangles,
                edges = global_edges,
                edge_markers = global_edge_markers,
                nodes4boundary = global_nodes4boundary)
            , batch_size=[2,])        
        
        return global_triangulation
    
    def build_global_triangulation(self, local_triangulations):
        global_vertices = []
        vertex_global_ids = []
        vertex_markers_global = []
        coord_to_global = {}  # clave: tuple(coord), exacto
        next_vertex_id = 0
    
        # Paso 1: construir lista global de vértices y mapas locales -> globales
        for local_mesh in torch.unbind(local_triangulations, dim = -1):
            local_vertices = local_mesh["vertices"]        # Tensor de forma (N_v, 2)
            local_vertex_markers = local_mesh["vertex_markers"]  # Tensor (N_v,)
            local_to_global = {}
    
            for local_id, coord in enumerate(local_vertices):
                key = tuple(coord.tolist())  # convertir a tuple para usar como clave
                marker = int(local_vertex_markers[local_id])

                if local_vertex_markers[local_id].item() >= 2:  # nodo en traza
                    if key in coord_to_global:
                        global_id = coord_to_global[key]
                    else:
                        global_id = next_vertex_id
                        coord_to_global[key] = global_id
                        global_vertices.append(coord)
                        next_vertex_id += 1
                    
                else:  # nodo exclusivo
                    global_id = next_vertex_id
                    coord_to_global[key] = global_id
                    global_vertices.append(coord)
                    next_vertex_id += 1
    
                    if marker == 1:
                        vertex_markers_global.append(global_id)
    
                local_to_global[local_id] = global_id
    
            vertex_global_ids.append(local_to_global)
    
        global_vertices = torch.stack(global_vertices, dim=0)
        vertex_markers_global = torch.tensor(vertex_markers_global)
    
        # Paso 2: reconstruir triángulos con índices globales
        global_triangles = []
        fracture_map = []
        trace_map = {}  # clave: traza k, valor: lista de (fractura_id, edge_id)                    

        for i, local_mesh in enumerate(local_triangulations):
            local_to_global = vertex_global_ids[i]
            local_triangles = local_mesh["triangles"]  # Tensor de shape (N_t, 3)
            edge_markers = local_mesh["edge_markers"]
            for tri in local_triangles:
                global_tri = [local_to_global[v.item()] for v in tri]
                global_triangles.append(torch.tensor(global_tri, dtype=torch.long))
                fracture_map.append(i)
            for edge_id in range(len(edge_markers)):
                marker = int(edge_markers[edge_id])
                if marker >= 2:  # traza
                    if marker not in trace_map:
                        trace_map[marker] = []
                    trace_map[marker].append((i, edge_id))
    
        global_triangles = torch.stack(global_triangles, dim=0)
        fracture_map = torch.tensor(fracture_map, dtype=torch.long)

        N = len(global_vertices)
        vertex_to_fracture = -torch.ones(N, dtype=torch.long)
    
        for fracture_id, local_to_global in enumerate(vertex_global_ids):
            for local_id, global_id in local_to_global.items():
                if vertex_to_fracture[global_id] == -1:
                    vertex_to_fracture[global_id] = fracture_id
    
        return {
            "vertices": global_vertices,            # (N_global_vertices, 2 o 3), float
            "triangles": global_triangles,          # (N_global_triangles, 3), long
            "fracture_map": fracture_map,           # (N_global_triangles,), long
            "trace_map": trace_map,                 # {k: [(fractura_id, edge_id), ...]}
            "vertex_global_ids": vertex_global_ids, # por fractura: local_id -> global_id
            "vertex_markers_global": vertex_markers_global,
            "vertex_to_fracture": vertex_to_fracture
        }
    
    

class Abstract_Element(ABC):
    def __init__(self,
                 P_order: int,
                 int_order: int):
        
        self.P_order = P_order
        self.int_order = int_order
        
        self.gaussian_nodes, self.gaussian_weights = self.compute_gauss_values()
        
    def compute_integral_values(self, coords4elements: torch.Tensor):
        
        det_map_jacobian, self.inv_map_jacobian = self.compute_map(coords4elements)
                
        bar_coords, v, v_grad = self.compute_shape_functions(self.gaussian_nodes, self.inv_map_jacobian)
                        
        integration_points = torch.split(bar_coords.mT @ coords4elements.unsqueeze(-3), 1, dim = -1)
                        
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

        integration_points = torch.concat(integration_points, dim = -1)        

        inv_map = (integration_points - first_node) @ inv_map_jacobian.mT
                
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
    
    def compute_det_and_inv_map(self, map_jacobian):

        det_map_jacobian = torch.linalg.norm(map_jacobian, dim = -2, keepdim = True)
        
        inv_map_jacobian = 1./det_map_jacobian
        
        return det_map_jacobian, inv_map_jacobian
    
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

    @staticmethod
    def compute_det_and_inv_map(map_jacobian: torch.Tensor):
        
        ab, cd = torch.split(map_jacobian, 1, dim = -2)
        
        a, b = torch.split(ab, 1, dim = -1)
        c, d = torch.split(cd, 1, dim = -1)

        det_map_jacobian = (a * d - b * c).unsqueeze(-3)
        
        inv_map_jacobian = (1 / det_map_jacobian) * torch.stack([torch.concat([d, -b], dim = -1),
                                                                  torch.concat([-c, a], dim = -1)], dim = -2)
    
        return det_map_jacobian, inv_map_jacobian
    
    
class Element_Fracture(Abstract_Element):
    def __init__(self, 
                 P_order, 
                 int_order):
        
        self.compute_barycentric_coordinates = lambda x: torch.stack([1.0 - x[..., [0]] - x[..., [1]], 
                                                                      x[..., [0]], 
                                                                      x[..., [1]]], dim = -2)
        
        self.barycentric_grad = torch.tensor([[-1.0, -1.0],
                                              [ 1.0,  0.0],
                                              [ 0.0,  1.0]])
        
        self.reference_element_area = 0.5
        
        # self.fracture_mapping_jacobian = torch.tensor([[-1., 1., 0.],
        #                                                [-1., 0., 1.]]) @ fractures_data[:,:3, :]
        
        # self.norm_fracture_mapping_jacobian = torch.norm(torch.linalg.cross(*torch.split(self.fracture_mapping_jacobian, 1, dim = -1), dim = -2), dim = -2, keepdim = True)
        
        super().__init__(P_order, 
                         int_order)
        
        def map_c4n_to_fracture(self, points):
            
            map_points =  points @ self.fractures_data[:,:3,:]
            
            return map_points        

        def map_to_fracture(self, points):
            
            map_points =  points @ self.fractures_data[:,:3,:]
            
            return map_points
        
        
    def compute_integral_values(self, coords4elements: torch.Tensor, fractures_data):
        
        fractures_vertices = fractures_data[:,:3,:].unsqueeze(-3).unsqueeze(-3)
        
        bar_coords = self.compute_barycentric_coordinates(self.gaussian_nodes) 
        
        mapp, det_map_jacobian, self.inv_map_jacobian = self.compute_map(coords4elements, bar_coords)
                
        fractures_map, fractures_map_jacobian, det_fractures_map_jacobian, fractures_map_jacobian_inv = self.compute_fracture_map(mapp, fractures_vertices)
        
        v, v_grad = self.compute_shape_functions(bar_coords, self.inv_map_jacobian, fractures_map_jacobian_inv)
                                
        integration_points = torch.split(fractures_map , 1, dim = -1)
                        
        dx = self.reference_element_area * self.gaussian_weights * det_map_jacobian * det_fractures_map_jacobian
        
        return v, v_grad, integration_points, dx, fractures_map_jacobian
    
    def compute_fracture_map (self, mapp, fractures_vertices):
        
        map_bar_coords = self.compute_barycentric_coordinates(mapp).squeeze(-3)
        
        fractures_map = map_bar_coords.mT @ fractures_vertices
        
        fractures_map_jacobian = fractures_vertices.mT @ self.barycentric_grad
        
        det_fractures_map_jacobian = torch.norm(torch.linalg.cross(*torch.split(fractures_map_jacobian, 1, dim = -1), dim = -2), dim = -2, keepdim = True)
        
        fractures_map_jacobian_inv = torch.linalg.inv(fractures_map_jacobian.mT @ fractures_map_jacobian) @ fractures_map_jacobian.mT 
        
        return fractures_map, fractures_map_jacobian, det_fractures_map_jacobian, fractures_map_jacobian_inv
    
    def compute_map(self, coords4elements: torch.Tensor, bar_coords: torch.Tensor):
        
        mapp = bar_coords.mT @ coords4elements.unsqueeze(-3)
        
        map_jacobian = coords4elements.mT @ self.barycentric_grad
        
        det_map_jacobian, inv_map_jacobian = self.compute_det_and_inv_map(map_jacobian)
        
        return mapp, det_map_jacobian, inv_map_jacobian
            
    def compute_shape_functions(self, bar_coords, inv_map_jacobian: torch.Tensor, fractures_map_jacobian_inv):
                
        v, v_grad = self.shape_functions_value_and_grad(bar_coords, inv_map_jacobian, fractures_map_jacobian_inv)

        return v, v_grad                 

    @staticmethod
    def compute_inverse_map(first_node: torch.Tensor, integration_points: torch.Tensor, inv_map_jacobian: torch.Tensor):

        integration_points = torch.concat(integration_points, dim = -1)        

        inv_map = (integration_points - first_node) @ inv_map_jacobian.mT
                
        return inv_map

    def shape_functions_value_and_grad(self, bar_coords: torch.Tensor, inv_map_jacobian: torch.Tensor, fractures_map_jacobian_inv):
                    
        v = bar_coords
        
        v_grad = self.barycentric_grad @ inv_map_jacobian.mT @ fractures_map_jacobian_inv
            
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

    @staticmethod
    def compute_det_and_inv_map(map_jacobian: torch.Tensor):
        
        ab, cd = torch.split(map_jacobian, 1, dim = -2)
        
        a, b = torch.split(ab, 1, dim = -1)
        c, d = torch.split(cd, 1, dim = -1)

        det_map_jacobian = (a * d - b * c).unsqueeze(-3)
        
        inv_map_jacobian = (1 / det_map_jacobian) * torch.stack([torch.concat([d, -b], dim = -1),
                                                                  torch.concat([-c, a], dim = -1)], dim = -2)
    
        return det_map_jacobian, inv_map_jacobian
    
class Abstract_Basis(ABC):
    def __init__(self,
                 mesh: Abstract_Mesh,
                 elements: Abstract_Element):
        
        
        
        self.elements = elements
        self.mesh = mesh
            
        self.v, self.v_grad, self.integration_points, self.dx,  = elements.compute_integral_values(mesh.coords4elements)
        
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
            new_nodes4dofs = edges_parameters["edges_idx"].reshape(mesh_parameters["nb_simplex"], 3) + mesh_parameters["nb_nodes"]
            new_nodes4boundary_dofs = edges_parameters["nodes_idx4boundary_edges"] + mesh_parameters["nb_nodes"]
            
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
                            "inner_dofs": inner_dofs}

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
            
        if basis.__class__ == Basis:
            
            elements_mask = self.mesh.map_fine_mesh(basis.mesh)
            
            dofs_idx = self.global_dofs4elements[elements_mask]
            
            coords4elements_first_node = self.coords4elements[..., [0], :][elements_mask]
        
            inv_map_jacobian = self.elements.inv_map_jacobian[elements_mask]

            new_integrations_points = self.elements.compute_inverse_map(coords4elements_first_node,
                                                                        basis.integration_points, 
                                                                        inv_map_jacobian)
            
            _, v, v_grad = self.elements.compute_shape_functions(new_integrations_points.squeeze(-2), inv_map_jacobian)

        if basis.__class__ == Interior_Facet_Basis: 

            elements_mask = basis.mesh.edges_parameters["elements4inner_edges"]
            
            dofs_idx = basis.mesh.nodes4elements[elements_mask]
                    
            coords4elements_first_node = self.coords4elements[..., [0], :][elements_mask]
        
            inv_map_jacobian = self.elements.inv_map_jacobian[elements_mask]
            
            integration_points = torch.split(torch.cat(basis.integration_points, dim = -1).unsqueeze(-4), 1, dim = -1)

            new_integrations_points = self.elements.compute_inverse_map(coords4elements_first_node,
                                                                        integration_points, 
                                                                        inv_map_jacobian)
            
            _, v, v_grad = self.elements.compute_shape_functions(new_integrations_points.squeeze(-3), inv_map_jacobian)
                
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

        self.elements = elements
        self.mesh = mesh
        
        nodes4elements = mesh.edges_parameters["nodes4inner_edges"]
        coords4nodes = mesh.coords4nodes
        coords4elements = mesh.coords4nodes[nodes4elements]
            
        self.v, self.v_grad, self.integration_points, self.dx = elements.compute_integral_values(coords4elements)
        
        self.coords4global_dofs, self.global_dofs4elements, self.nodes4boundary_dofs = self.compute_dofs(coords4nodes, 
                                                                                                         nodes4elements, 
                                                                                                         mesh.nodes4boundary,
                                                                                                         mesh.mesh_parameters,
                                                                                                         mesh.edges_parameters,
                                                                                                         elements.P_order)

        self.coords4elements = self.coords4global_dofs[self.global_dofs4elements]

        self.basis_parameters = self.compute_basis_parameters(self.coords4global_dofs, 
                                                              self.global_dofs4elements, 
                                                              self.nodes4boundary_dofs)
        
    def compute_dofs(self, coords4nodes, nodes4elements, nodes4boundary, mesh_parameters, edges_parameters, P_order):
        
        if P_order == 1:
            coords4global_dofs = coords4nodes
            global_dofs4elements = nodes4elements
            nodes4boundary_dofs = nodes4boundary
            
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