from torch_fem import MeshTri
import triangle
import torch
from matplotlib import pyplot as plt

mesh_data = triangle.triangulate(
    {"vertices": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]},
    "qena" + str(0.5**4),
)

fig, axis = plt.subplots()

triangle.plot(axis, **mesh_data)

mesh = MeshTri(triangulation=mesh_data)

interior_nodes = torch.nonzero(
    mesh["vertices", "markers"].squeeze(-1) != 1, as_tuple=True
)[0]

node_to_triangle_map = mesh.build_node_to_triangle_map(
    mesh["cells", "vertices"], interior_nodes
)
print(node_to_triangle_map)

plt.show()
