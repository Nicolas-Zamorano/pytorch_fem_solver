import torch

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import PolyCollection
import numpy as np
import triangle as tr
from typing import Dict, Tuple

from torch_fem import MeshTri, ElementTri, Basis

# torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
# torch.cuda.empty_cache()
torch.set_default_dtype(torch.float64)

vertices = [
    [0.0, 0.0],  # 0
    [1.0, 0.0],  # 1
    [1.0, 1.0],  # 2
    [0.0, 1.0],  # 3
]
segments = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],
]


def refine_to_p3(mesh: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Uniformly refine a triangular mesh so that:
      - new vertices are exactly the P3 Lagrange nodes (vertices, 1/3 and 2/3 edge points, and the single interior point),
      - each coarse triangle is split into 9 small triangles (no hanging nodes).
    Input/Output follow 'triangle' dict format: {'vertices': (N,2), 'triangles': (M,3)}.
    """
    V = np.asarray(mesh["vertices"], dtype=np.float64)
    T = np.asarray(mesh["triangles"], dtype=np.int32)
    n = 3  # P3 lattice granularity

    new_vertices: list = V.tolist()  # start with coarse vertices
    new_triangles: list = []

    # Map (emin, emax, t) -> global index, where t in {1,2} encodes 1/3 and 2/3 from min->max
    edge_nodes: Dict[Tuple[int, int, int], int] = {}

    def get_edge_node_idx(u: int, v: int, j: int) -> int:
        """
        Get/create the node at j/3 along directed edge u->v, deduped across triangles.
        Stored w.r.t. the undirected edge (emin->emax) at t in {1,2}.
        """
        emin, emax = (u, v) if u < v else (v, u)
        tnum = j if u == emin else n - j  # normalize to (emin->emax)
        key = (emin, emax, tnum)
        if key in edge_nodes:
            return edge_nodes[key]
        ta = tnum / float(n)
        p = (1.0 - ta) * V[emin] + ta * V[emax]
        idx = len(new_vertices)
        new_vertices.append(p.tolist())
        edge_nodes[key] = idx
        return idx

    for a, b, c in T:
        va, vb, vc = V[a], V[b], V[c]

        # Local mapping from barycentric integer triple (i,j,k) with i+j+k=3 -> global vertex index
        local: Dict[Tuple[int, int, int], int] = {}

        # Vertices
        local[(3, 0, 0)] = int(a)
        local[(0, 3, 0)] = int(b)
        local[(0, 0, 3)] = int(c)

        # Edge nodes at 1/3 and 2/3
        # AB edge: (2,1,0), (1,2,0)
        local[(2, 1, 0)] = get_edge_node_idx(int(a), int(b), 1)
        local[(1, 2, 0)] = get_edge_node_idx(int(a), int(b), 2)
        # BC edge: (0,2,1), (0,1,2)  => parameter = k/3 from b->c
        local[(0, 2, 1)] = get_edge_node_idx(int(b), int(c), 1)
        local[(0, 1, 2)] = get_edge_node_idx(int(b), int(c), 2)
        # CA edge: (1,0,2), (2,0,1)  => parameter = i/3 from c->a
        local[(1, 0, 2)] = get_edge_node_idx(int(c), int(a), 1)
        local[(2, 0, 1)] = get_edge_node_idx(int(c), int(a), 2)

        # Interior node (1,1,1)
        center = (va + vb + vc) / 3.0
        center_idx = len(new_vertices)
        new_vertices.append(center.tolist())
        local[(1, 1, 1)] = center_idx

        def idx(i: int, j: int, k: int) -> int:
            return local[(i, j, k)]

        # Assemble 9 sub-triangles using barycentric grid (Up and Down patterns)
        # Up triangles: i=0..n-1, j=0..n-1-i
        for i in range(n):
            for j in range(n - i):
                # p1: (i+1, j, n - i - j - 1), p2: (i, j+1, n - i - j - 1), p3: (i, j, n - i - j)
                p1 = idx(i + 1, j, n - i - j - 1)
                p2 = idx(i, j + 1, n - i - j - 1)
                p3 = idx(i, j, n - i - j)
                new_triangles.append([p1, p2, p3])

        # Down triangles: i=0..n-2, j=0..n-2-i
        for i in range(n - 1):
            for j in range(n - 1 - i):
                # p1: (i+1, j+1, n - i - j - 2), p2: (i+1, j, n - i - j - 1), p3: (i, j+1, n - i - j - 1)
                p1 = idx(i + 1, j + 1, n - i - j - 2)
                p2 = idx(i + 1, j, n - i - j - 1)
                p3 = idx(i, j + 1, n - i - j - 1)
                new_triangles.append([p1, p2, p3])

    return {
        "vertices": np.asarray(new_vertices, dtype=np.float64),
        "triangles": np.asarray(new_triangles, dtype=np.int32),
    }

# Triangulación inicial
mesh_data_coarser = tr.triangulate({"vertices": vertices}, "pelqna" + str(0.5**5))

# mesh_coarser = MeshTri(triangulation=mesh_data_coarser)

# elements_coarser = ElementTri(polynomial_order=3, integration_order=2)

# basis_coarser = Basis(mesh_coarser, elements_coarser)

# new_vertices = basis_coarser._coords4global_dofs.numpy(force=True)


# Triangulación inicial
mesh_data_coarser = tr.triangulate({"vertices": vertices}, "qa" + str(0.5**1))

# Ensure correct dtypes
mesh_data_coarser["vertices"] = np.asarray(
    mesh_data_coarser["vertices"], dtype=np.float64
)
mesh_data_coarser["triangles"] = np.asarray(
    mesh_data_coarser["triangles"], dtype=np.int32
)

# P3 refinement with no hanging nodes (9 sub-triangles per coarse triangle)
mesh_data_finer = refine_to_p3(mesh_data_coarser)

tr.compare(
    plt,
    mesh_data_coarser,
    mesh_data_finer,
)

plt.show()

# mesh_finer = MeshTri(triangulation=mesh_data_finer)

# elements_finer = ElementTri(polynomial_order=1, integration_order=2)

# basis_finer = Basis(mesh_finer, elements_finer)


# mapped_basis = mesh_coarser.map_fine_mesh(mesh_finer)

# fig_mapped, ax_mapped = plt.subplots()

# c4n_coarser = torch.Tensor.numpy(mesh_coarser["cells", "coordinates"], force=True)
# c4n_finer = torch.Tensor.numpy(mesh_finer["cells", "coordinates"], force=True)

# # Plot coarser mesh (cyan with red edges)
# ax_mapped.add_collection(
#     PolyCollection(
#         c4n_coarser, facecolors="cyan", linewidths=1.5, edgecolors="r", alpha=0.25
#     )
# )

# # Plot finer mesh (magenta with blue edges)
# ax_mapped.add_collection(
#     PolyCollection(
#         c4n_finer, facecolors="magenta", linewidths=0.5, edgecolors="b", alpha=0.25
#     )
# )

# # Compute centroids of fine mesh triangles
# c4n_finer_centroids = c4n_finer.mean(axis=1)

# # Annotate each fine triangle with its mapped coarse triangle index
# for i, (x, y) in enumerate(c4n_finer_centroids):
#     mapped_idx = mapped_basis[i].item()
#     ax_mapped.text(
#         x, y, str(mapped_idx), ha="center", va="center", fontsize=8, color="black"
#     )

# ax_mapped.set_aspect("equal")
# ax_mapped.set_xlabel("x")
# ax_mapped.set_ylabel("y")
# ax_mapped.set_title("Fine mesh (magenta) mapped to coarse mesh (cyan)")

# plt.show()
