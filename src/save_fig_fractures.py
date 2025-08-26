import torch

import tensordict as td
import triangle as tr
import numpy as np

from fracture_fem import Fractures

torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
torch.set_default_dtype(torch.float64)

#---------------------- FEM Parameters ----------------------#

h = 0.5

n = 10

fracture_2d_data = {"vertices" : [[-1., 0.],
                                  [ 1., 0.],
                                  [-1., 1.],
                                  [ 1., 1.],
                                  [ 0., 0.],
                                  [ 0., 1.]],
                    "segments" : [[0, 2],
                                  [0, 4],
                                  [1, 3],
                                  [1, 4],
                                  [2, 5],
                                  [3, 5],
                                  [4, 5]],
                    "segment_markers": [[1],
                                        [1],
                                        [1],
                                        [1],
                                        [1],
                                        [1],
                                        [0]]
                    }

fracture_triangulation = tr.triangulate(fracture_2d_data, 
                                                      "pqsea"+str(h**n)
                                                      )

fracture_triangulation_torch = td.TensorDict(fracture_triangulation)

fractures_triangulation = (fracture_triangulation_torch, fracture_triangulation_torch)



fractures_data = torch.tensor([[[-1., 0., 0.], 
                                [ 1., 0., 0.], 
                                [-1., 1., 0.], 
                                [ 1., 1., 0.]],
                               
                               [[ 0., 0.,-1.], 
                                [ 0., 0., 1.], 
                                [ 0., 1.,-1.], 
                                [ 0., 1., 1.]]])

mesh = Fractures(triangulations = fractures_triangulation,
                 fractures_3D_data = fractures_data)




vertices = mesh.local_triangulations["vertices_3D"]
triangles = mesh.local_triangulations["triangles"]
vertices_2D = mesh.local_triangulations["vertices"]

vertices_fracture_1, vertices_fracture_2 = torch.unbind(vertices_2D, dim = 0)
triangles_fracture_1, triangles_fracture_2 = torch.unbind(triangles, dim = 0)


def exact(x, y, z):
    
    x_fracture_1, x_fracture_2 = torch.split(x, 1, dim = 0)
    y_fracture_1, y_fracture_2 = torch.split(y, 1, dim = 0)
    z_fracture_1, z_fracture_2 = torch.split(z, 1, dim = 0)

    exact_fracture_1 = -y_fracture_1 * (1 - y_fracture_1) * torch.abs(x_fracture_1) * (x_fracture_1**2 - 1)
    exact_fracture_2 =  y_fracture_2 * (1 - y_fracture_2) * torch.abs(z_fracture_2) * (z_fracture_2**2 - 1)

    
    exact_value = torch.cat([exact_fracture_1, exact_fracture_2], dim = 0)

    return exact_value

exact_eval = exact(*torch.unbind(vertices, dim = -1))

exact_eval_fracture_1, exact_eval_fracture_2 = torch.unbind(exact_eval, dim = 0)

import pyvista as pv
import os
import numpy as np
import torch

save_dir = "figures"
name = "exact"
os.makedirs(save_dir, exist_ok=True)

plotter = pv.Plotter(off_screen=True)
colors = ["red", "blue"]

for i in range(2):  # Para cada fractura
    verts = vertices[i].cpu().numpy()  # (N_v, 3)
    tris = triangles[i].cpu().numpy()  # (N_T, 3)
    sol = exact_eval[i].cpu().numpy() 

    # Crear caras para pyvista
    faces = np.hstack([np.full((tris.shape[0], 1), 3), tris]).flatten()
    mesh = pv.PolyData(verts, faces)

    # Añadir la malla con color específico
    plotter.add_mesh(
        mesh,
        show_edges=True,
        # color=colors[i],
        opacity=1.0,
        lighting=False,
        label=f"Fracture {i+1}"
    )
    
    # Añadir etiquetas visibles en la escena (usa el primer vértice para colocar el texto)
    plotter.add_point_labels(
        verts[[0]],  # posición de la etiqueta (puedes cambiar a otro punto si prefieres)
        [f"Fracture {i+1}"],
        # text_color=colors[i],
        font_size=12,
        point_color=colors[i],
        point_size=10,
        render_points_as_spheres=True,
        always_visible=True
    )

plotter.show_grid()
plotter.show()
plotter.screenshot(os.path.join(save_dir, "domain_simple.png"))