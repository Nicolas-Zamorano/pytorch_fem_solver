import torch

import tensordict as td
import triangle as tr
import pyvista as pv

torch.set_default_dtype(torch.float64)

def stack_triangulations(fracture_triangulations: tuple):
    
    local_vertices = torch.stack([triangulation["vertices"] for triangulation in fracture_triangulations], dim = 0)
    local_vertices_marker = torch.stack([triangulation["vertex_markers"] for triangulation in fracture_triangulations], dim = 0)
    
    local_triangles = torch.stack([triangulation["triangles"] for triangulation in fracture_triangulations], dim = 0)
    
    local_edges = torch.stack([triangulation["edges"] for triangulation in fracture_triangulations], dim = 0)
    local_edges_marker = torch.stack([triangulation["edge_markers"] for triangulation in fracture_triangulations], dim = 0)

    local_triangulations = td.TensorDict(
        vertices = local_vertices,
        vertex_markers = local_vertices_marker,
        triangles = local_triangles,
        edges = local_edges,
        edge_markers = local_edges_marker,
        )
              
    return local_triangulations

def compute_barycentric_coordinates(x):
    
    return torch.stack([1.0 - x[..., [0]] - x[..., [1]], 
                        x[..., [0]], 
                        x[..., [1]]], dim = -2)

def build_global_triangulation(local_triangulations):
    
    local_triangulation_bar_coords = compute_barycentric_coordinates(local_triangulations["vertices"])
    
    nb_fractures, nb_vertices, nb_dim = local_triangulations["vertices"].shape
    
    nb_edges = local_triangulations["edges"].shape[-2]

    local_triangulation_3D_coords = (local_triangulation_bar_coords.mT @  fractures_3D_vertices).reshape(-1,3)

    global_vertices_3D, global2local_idx = torch.unique(local_triangulation_3D_coords, 
                                                        dim = 0,
                                                        return_inverse = True,
                                                        )
    
    nb_global_vertices = global_vertices_3D.shape[-2]
    
    local2global_idx = torch.full((nb_global_vertices,), (nb_fractures*nb_vertices)+1, dtype = torch.int64)
    
    local2global_idx.scatter_reduce_(0, 
                                     global2local_idx, 
                                     torch.arange(nb_fractures*nb_vertices), 
                                     reduce = "amin",
                                     include_self = True)
    
    vertices_offset = torch.arange(nb_fractures)[:, None, None] * nb_vertices
    global_triangles = global2local_idx[local_triangulations["triangles"] + vertices_offset].reshape(-1,3)

    local_edges_2_global = global2local_idx[local_triangulations["edges"] + vertices_offset].reshape(-1,2)
    
    global_edges, global2local_edges_idx = torch.unique(local_edges_2_global.reshape(-1, 2), 
                                                        dim = 0,
                                                        return_inverse = True)
    
    nb_global_edges = global_edges.shape[-2]
    
    local2global_edges_idx = torch.full((nb_global_edges,), (nb_fractures*nb_edges)+1, dtype = torch.int64)
    
    local2global_edges_idx.scatter_reduce_(0, 
                                           global2local_edges_idx, 
                                           torch.arange(nb_fractures*nb_edges), 
                                           reduce = "amin",
                                           include_self = True)
    
    global_vertices_marker = local_triangulations["vertex_markers"].reshape(-1)[local2global_idx]
    global_edges_marker = local_triangulations["edge_markers"].reshape(-1)[local2global_edges_idx]

    global_triangulation = td.TensorDict(vertices = global_vertices_3D,
                                         vertex_markers = global_vertices_marker,
                                         triangles = global_triangles,
                                         edges = global_edges,
                                         edge_markers = global_edges_marker,
                                         global2local_idx = global2local_idx,
                                         local2global_idx = local2global_idx)
    
    return global_triangulation

def exact(x, y, z):
    
    x_fracture_1, x_fracture_2 = torch.split(x, 1, dim = 0)
    y_fracture_1, y_fracture_2 = torch.split(y, 1, dim = 0)
    z_fracture_1, z_fracture_2 = torch.split(z, 1, dim = 0)

    exact_fracture_1 = -y_fracture_1 * (1 - y_fracture_1) * torch.abs(x_fracture_1) * (x_fracture_1**2 - 1)
    exact_fracture_2 =  y_fracture_2 * (1 - y_fracture_2) * torch.abs(z_fracture_2) * (z_fracture_2**2 - 1)
    
    exact_value = torch.cat([exact_fracture_1, exact_fracture_2], dim = 0)

    return exact_value

# def exact(x, y, z):
    
#     x_fracture_1, x_fracture_2 = torch.split(x, 1, dim = 0)
#     y_fracture_1, y_fracture_2 = torch.split(y, 1, dim = 0)
#     z_fracture_1, z_fracture_2 = torch.split(z, 1, dim = 0)

#     rhs_fracture_1 =  6. * (y_fracture_1 - y_fracture_1**2) * torch.abs(x_fracture_1) - 2. * (torch.abs(x_fracture_1)**3 - torch.abs(x_fracture_1))
#     rhs_fracture_2 = -6. * (y_fracture_2 - y_fracture_2**2) * torch.abs(z_fracture_2) + 2. * (torch.abs(z_fracture_2)**3 - torch.abs(z_fracture_2))
    
#     rhs_value = torch.cat([rhs_fracture_1, rhs_fracture_2], dim = 0)

#     return rhs_value

h = 0.5**5

fracture_2d_data = {"vertices" : [[0., 0.],
                                  [.5, 0.],
                                  [1., 0.],
                                  [.5, .5],
                                  [0., 1.],
                                  [.5, 1.],
                                  [1., 1.]],
                    "segments" : [[0, 1],
                                  [0, 4],
                                  [1, 2],
                                  [1, 3],
                                  [2, 6],
                                  [3, 5],
                                  [4, 5],
                                  [5, 6]]
                    }

fracture_triangulation = td.TensorDict(tr.triangulate(fracture_2d_data, 
                                                      "pqsena"+str(h)
                                                      ))

# tr.compare(plt, fracture_2d_data, fracture_triangulation)

fractures_triangulation = (fracture_triangulation, fracture_triangulation)

fractures_3D_data = torch.tensor(
    [[[-1., 0., 0.], 
     [ 1., 0., 0.], 
     [-1., 1., 0.], 
     [ 1., 1., 0.]],                            
    [[ 0., 0.,-1.], 
     [ 0., 0., 1.], 
     [ 0., 1.,-1.], 
     [ 0., 1., 1.]]]
    )

fractures_3D_vertices = fractures_3D_data[:, :3, :].unsqueeze(-3)

local_triangulations = stack_triangulations(fractures_triangulation)

global_triangulation = build_global_triangulation(local_triangulations)

local_vertices_3D = global_triangulation["vertices"][global_triangulation["global2local_idx"].reshape(2, -1)] 

vertices = global_triangulation["vertices"].numpy()

exact_value_local = exact(*torch.split(local_vertices_3D, 1, -1))

exact_value_global = exact_value_local.reshape(-1,1)[global_triangulation["local2global_idx"]]

faces = torch.cat([torch.full(((global_triangulation["triangles"].shape[0], 1)), 3), global_triangulation["triangles"]], dim = -1).reshape(-1).numpy()

mesh = pv.PolyData(vertices, faces)

mesh.point_data["u"] = exact_value_global.numpy()

plotter = pv.Plotter()
plotter.add_mesh(mesh, scalars = "u", show_edges=True, color='lightblue')
plotter.show()



