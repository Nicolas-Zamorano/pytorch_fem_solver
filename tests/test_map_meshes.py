import torch

import matplotlib.pyplot as plt
import triangle as tr

from torch_fem import MeshTri, ElementTri, Basis

# torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
# torch.cuda.empty_cache()
torch.set_default_dtype(torch.float64)

vertices = [
    [0.0, 0.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [0.0, 1.0],
]
segments = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],
]

mesh_data_coarser = tr.triangulate({"vertices": vertices}, "eqna" + str(0.5**1))

mesh_coarser = MeshTri(triangulation=mesh_data_coarser)

elements_coarser = ElementTri(polynomial_order=2, integration_order=4)

basis_coarser = Basis(mesh_coarser, elements_coarser)

new_vertices = basis_coarser.coords4global_dofs.numpy(force=True)

new_segments = basis_coarser.vertices_4_new_edges.numpy(force=True)

centroids = mesh_coarser["cells", "coordinates"].mean(dim=-2).numpy(force=True)

regions = [[c[0], c[1], i, 0] for i, c in enumerate(centroids)]

mesh_data_finer = tr.triangulate(
    {"vertices": new_vertices, "segments": new_segments, "regions": regions},
    "penA",
)

tr.compare(
    plt,
    mesh_data_coarser,
    mesh_data_finer,
)

plt.show()
