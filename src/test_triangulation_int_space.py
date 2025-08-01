import triangle as tr
import matplotlib.pyplot as plt
import numpy as np

h = 0.5**1

xd = tr.get_data("LA")

vertices = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]

convex_hull = tr.convex_hull(vertices)

mesh_H = tr.triangulate(
    dict(vertices=vertices, segments=convex_hull), "pcqea" + (str(h))
)

c4e = mesh_H["vertices"][mesh_H["triangles"]]

centroids = c4e[:, :].mean(-2)

# centroids[0,0] +=0.1
# centroids[1,0] +=0.1

regions = np.concatenate(
    [
        np.round(centroids, 3),
        np.expand_dims(np.arange(c4e.shape[0]), -1),
        np.zeros((c4e.shape[0], 1)),
    ],
    axis=-1,
)

mesh_H["regions"] = regions

mesh_h = tr.triangulate(mesh_H, "pra" + str(h / 2))

mesh_xd = tr.triangulate(mesh_h, "pA+aqW")

mesh_h = tr.triangulate(
    dict(vertices=vertices, segments=mesh_H["edges"], regions=regions),
    "DqspAa" + str(0.25 * h),
)

tr.compare(plt, mesh_H, mesh_h)

plt.show()
