import torch
import math

import matplotlib.pyplot as plt
import tensordict as td
import triangle as tr

h = 0.5 ** (4)

vertices = [[0.0, 0.0], [1, 0.0], [0.0, 1.0], [1.0, 1.0]]

vertex_markers = [[1], [2], [2], [1]]

triangles = [[0, 1, 2], [1, 2, 3]]

# segments = [[0, 1],
#             [0, 4],
#             [1, 2],
#             [1, 3],
#             [2, 6]]

# segment_markers =[[1],
#                   [1],
#                   [1],
#                   [2],
#                   [1],
#                   [2],
#                   [1],
#                   [1]]

reference_fracture_data = dict(
    vertices=vertices,
    triangles=triangles,
    vertex_markers=vertex_markers,
    # segments = segments,
    # segment_markers = segment_markers
)

reference_fracture_mesh = tr.triangulate(reference_fracture_data, "")

ax = plt.subplot()

tr.plot(ax, **reference_fracture_mesh)

plt.show()
