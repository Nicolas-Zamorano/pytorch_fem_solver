import torch
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from torch_fem import Patches, ElementTri, PatchesBasis

centers = torch.tensor([[0.5, 0.5]])

radius = torch.tensor([[0.5]])

# centers = torch.tensor([[0.25, 0.25], [0.25, 0.75], [0.75, 0.75], [0.75, 0.25]])

# radius = torch.tensor([[0.25], [0.25], [0.25], [0.25]])

centers = torch.tensor(
    [
        [0.125, 0.125],
        [0.125, 0.375],
        [0.125, 0.625],
        [0.125, 0.875],
        [0.375, 0.125],
        [0.375, 0.375],
        [0.375, 0.625],
        [0.375, 0.875],
        [0.625, 0.125],
        [0.625, 0.375],
        [0.625, 0.625],
        [0.625, 0.875],
        [0.875, 0.125],
        [0.875, 0.375],
        [0.875, 0.625],
        [0.875, 0.875],
    ],
    requires_grad=False,
)

radius = torch.tensor([[0.125]] * 16, requires_grad=False)


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
