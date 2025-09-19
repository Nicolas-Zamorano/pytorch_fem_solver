"""Test the global to local indexing for a set of 2 fractures in 3D."""

import torch

import triangle as tr
import pyvista as pv

from torch_fem import FracturesTri, FractureBasis, ElementTri

torch.set_default_dtype(torch.float64)


def exact(x, y, z):
    """Exact solution on the 2 fractures."""
    x_fracture_1, _ = torch.split(x, 1, dim=0)
    y_fracture_1, y_fracture_2 = torch.split(y, 1, dim=0)
    _, z_fracture_2 = torch.split(z, 1, dim=0)

    exact_fracture_1 = (
        -y_fracture_1
        * (1 - y_fracture_1)
        * torch.abs(x_fracture_1)
        * (x_fracture_1**2 - 1)
    )
    exact_fracture_2 = (
        y_fracture_2
        * (1 - y_fracture_2)
        * torch.abs(z_fracture_2)
        * (z_fracture_2**2 - 1)
    )

    exact_value = torch.cat([exact_fracture_1, exact_fracture_2], dim=0)

    return exact_value


# def exact(x, y, z):
#     """Exact solution on the 2 fractures."""
#     x_fracture_1, _ = torch.split(x, 1, dim=0)
#     y_fracture_1, y_fracture_2 = torch.split(y, 1, dim=0)
#     _, z_fracture_2 = torch.split(z, 1, dim=0)

#     rhs_fracture_1 = 6.0 * (y_fracture_1 - y_fracture_1**2) * torch.abs(
#         x_fracture_1
#     ) - 2.0 * (torch.abs(x_fracture_1) ** 3 - torch.abs(x_fracture_1))
#     rhs_fracture_2 = -6.0 * (y_fracture_2 - y_fracture_2**2) * torch.abs(
#         z_fracture_2
#     ) + 2.0 * (torch.abs(z_fracture_2) ** 3 - torch.abs(z_fracture_2))

#     rhs_value = torch.cat([rhs_fracture_1, rhs_fracture_2], dim=0)

#     return rhs_value


MESH_SIZE = 0.5**1

fracture_2d_data = {
    "vertices": [
        [-1.0, 0.0],
        [1.0, 0.0],
        [-1.0, 1.0],
        [1.0, 1.0],
        [0.0, 0.0],
        [0.0, 1.0],
    ],
    "segments": [[0, 2], [0, 4], [1, 3], [1, 4], [2, 5], [3, 5], [4, 5]],
    "segment_markers": [[1], [1], [1], [1], [1], [1], [0]],
}

fracture_triangulation = tr.triangulate(fracture_2d_data, "pqsena" + str(MESH_SIZE))


fractures_triangulation = [fracture_triangulation, fracture_triangulation]

fractures_data = torch.tensor(
    [
        [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        [[0.0, 0.0, -1.0], [0.0, 0.0, 1.0], [0.0, 1.0, -1.0], [0.0, 1.0, 1.0]],
    ]
)

fractures = FracturesTri(fractures_triangulation, fractures_data)

V = FractureBasis(fractures, ElementTri(1, 2))

global_triangulations = V.global_triangulation

local_vertices_3D = global_triangulations["vertices_3D"][
    global_triangulations["global2local_idx"].reshape(2, -1)
]

vertices = global_triangulations["vertices_3D"].numpy()

exact_value_local = exact(*torch.split(local_vertices_3D, 1, -1))

exact_value_global = exact_value_local.reshape(-1, 1)[
    global_triangulations["local2global_idx"]
]

faces = (
    torch.cat(
        [
            torch.full(((global_triangulations["triangles"].shape[0], 1)), 3),
            global_triangulations["triangles"],
        ],
        dim=-1,
    )
    .reshape(-1)
    .numpy()
)

mesh = pv.PolyData(vertices, faces)

mesh.point_data["u"] = exact_value_global.numpy()

plotter = pv.Plotter()
plotter.add_mesh(mesh, scalars="u", show_edges=True, color="lightblue")
plotter.show()
