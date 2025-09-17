"""Test the mapping from 2D to 3D for fractures."""

import torch
import tensordict as td
import triangle as tr
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)

MESH_SIZE = 0.5**1

fracture_2d_data = {
    "vertices": [
        [-1.0, 0.0],
        [1.0, 0.0],
        [-1.0, 1.0],
        [1.0, 1.0],
        [0.0, 0.0],
        [0.0, 0.5],
        [0.0, 1.0],
    ],
    "segments": [[0, 1], [1, 3], [2, 3], [0, 2], [4, 5], [5, 6]],
}

fracture_triangulation = td.TensorDict(
    tr.triangulate(fracture_2d_data, "pqsena" + str(MESH_SIZE))
)

fracture_2D_vertices = fracture_triangulation["vertices"][:3]

# fracture_3D_vertices = torch.tensor( [[-1., 0., 0.],
#                                       [ 1., 0., 0.],
#                                       [-1., 1., 0.]])

fracture_3D_vertices = torch.tensor(
    [[0.0, 0.0, -1.0], [0.0, 0.0, 1.0], [0.0, 1.0, -1.0]]
)

hat_V = torch.cat([fracture_2D_vertices.T, torch.ones(1, 3)], dim=0)
V = fracture_3D_vertices.T  # (3,3)

T = V @ torch.inverse(hat_V)  # (3,3)

A = T[..., :2]  # (3x2)
b = T[..., [-1]]  # (3x1)


def fracture_map(points_2d):
    """Map 2D coordinates to 3D using an affine transformation defined by matrix A and vector b."""
    return (A @ points_2d.mT + b).mT  # Output: (N, 3)


vertices_3D = fracture_map(fracture_triangulation["vertices"])

triangles = fracture_triangulation["triangles"]

x, y, z = vertices_3D[:, 0], vertices_3D[:, 1], vertices_3D[:, 2]

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.plot_trisurf(x, y, triangles, z, cmap="viridis", edgecolor="k")

ax.set_box_aspect([1, 1, 1])
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.tight_layout()
plt.show()
