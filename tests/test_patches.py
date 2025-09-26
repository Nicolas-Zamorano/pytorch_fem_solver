import torch
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from torch_fem import Patches, ElementTri, PatchesBasis


def generate_patches_info(n):
    """generates a set of centers and radius"""
    initial_centers = [(0.5, 0.5)]
    initial_radius = [0.5]

    for _ in range(n):
        new_centers = []
        new_radius = []
        for (cx, cy), r in zip(initial_centers, initial_radius):
            new_r = r / 2
            new_centers.extend(
                [
                    (cx - new_r, cy - new_r),
                    (cx - new_r, cy + new_r),
                    (cx + new_r, cy - new_r),
                    (cx + new_r, cy + new_r),
                ]
            )
            new_radius.extend([new_r] * 4)
        initial_centers, initial_radius = new_centers, new_radius

    return torch.Tensor(initial_centers), torch.Tensor(initial_radius).unsqueeze(-1)


centers, radius = generate_patches_info(3)


patches = Patches(centers, radius)

V = PatchesBasis(patches, ElementTri(1, 2))

c4e4p = patches["cells", "coordinates"]

colors = ["r", "g", "b", "m", "c", "y", "k"]
fig, ax = plt.subplots()
N_E = c4e4p.size(0)

cmp = plt.get_cmap("hsv", N_E)

for i, c4e in enumerate(c4e4p):
    triangle = c4e.numpy(force=True)
    poly = PolyCollection(triangle, facecolors="none", edgecolors=cmp(i), linewidths=1)
    ax.add_collection(poly)

ax.set_aspect("equal")
ax.autoscale_view()
plt.show()
