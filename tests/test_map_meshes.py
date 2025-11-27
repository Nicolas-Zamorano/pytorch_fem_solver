"""Test mesh mapping from coarse with higher degree to fine mesh with lower polynomial degree."""

import torch

import matplotlib.pyplot as plt
import triangle as tr

from torch_fem import MeshTri, ElementTri, Basis, ElementLine, InteriorEdgesBasis

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

elements_edges_coarser = ElementLine(polynomial_order=2, integration_order=4)

basis_edges_coarser = InteriorEdgesBasis(mesh_coarser, elements_edges_coarser)

new_vertices = basis_coarser.coordinates_4_global_dofs.numpy(force=True)

new_vertices_markers = basis_coarser.nodes_4_boundary_dofs.numpy(force=True)

new_edges_markers = basis_edges_coarser.markers_4_new_edges.numpy(force=True)

new_segments = basis_coarser.vertices_4_new_edges.numpy(force=True)

segments_markers = basis_coarser.enumeration_4_edges.numpy(force=True)


centroids = torch.Tensor.numpy(
    mesh_coarser["cells", "coordinates"].mean(dim=-2), force=True
)

regions = [[c[0], c[1], i, 0] for i, c in enumerate(centroids)]

mesh_data_finer = tr.triangulate(
    {
        "vertices": new_vertices,
        "segments": new_segments,
        "regions": regions,
        "segment_markers": segments_markers,
        "vertex_markers": new_vertices_markers,
        "edge_markers": new_edges_markers,
    },
    "penA",
)

print(mesh_data_finer)

tr.compare(
    plt,
    mesh_data_coarser,
    mesh_data_finer,
)

plt.show()
