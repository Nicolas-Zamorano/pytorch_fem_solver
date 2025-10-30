"# Example of solving a Poisson equation using a neural network and FEM basis functions."

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import torch
import triangle as tr

from torch_fem import (
    Basis,
    ElementTri,
    MeshTri,
    ElementLine,
    InteriorEdgesBasis,
    FeedForwardNeuralNetwork as NeuralNetwork,
    Model,
    DistanceFunctionBC,
)

# pyright: reportCallIssue=false

torch.set_default_dtype(torch.float64)

mesh_data = tr.triangulate(
    {"vertices": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]},
    "qena" + str(0.5**10),
)


mesh = MeshTri(triangulation=mesh_data)

elements = ElementTri(polynomial_order=1, integration_order=1)

discrete_basis = Basis(mesh, elements)


def exact(coordinates: torch.Tensor) -> torch.Tensor:
    """Exact solution of the PDE."""
    x, _ = torch.split(coordinates, 1, -1)
    # return torch.tanh(2 * (x**3 - y**4))
    return torch.ones_like(x)


exact_value = exact(discrete_basis.coords_4_global_dofs)

indices_4_dofs = discrete_basis.global_dofs_4_elements.unsqueeze(-2)

global_dofs_4_elements = discrete_basis.global_dofs_4_elements

xd = (exact_value[indices_4_dofs] * discrete_basis.v).sum(dim=-2).mean(-2)

v = discrete_basis.v.repeat(global_dofs_4_elements.shape[0], 1, 1, 1)
v_grad = discrete_basis.v_grad.repeat(1, v.shape[-3], 1, 1)


markers_4_dofs = discrete_basis.nodes_4_boundary_dofs.squeeze(-1)

exact_value[markers_4_dofs == 1] = 0.0

markers_4_elements = (
    markers_4_dofs[discrete_basis.global_dofs_4_elements]
    .unsqueeze(-2)
    .repeat(1, v.shape[-3], 1)
)


# v[markers_4_elements != 1, :] = 0.0
# v_grad[markers_4_elements == 1, :] = 0.0

interpolation = torch.sum(exact_value[indices_4_dofs] * v, dim=-2, keepdim=True).mean(
    dim=-2, keepdim=True
)


figure_solution, axis_solution = plt.subplots()

c4e = torch.Tensor.numpy(mesh["cells", "coordinates"], force=True)


triangles_plot = PolyCollection(
    c4e,  # type: ignore
    array=interpolation.reshape(-1),
    cmap="viridis",
    edgecolors="black",
    linewidths=0.2,
)

axis_solution.add_collection(triangles_plot)
axis_solution.autoscale_view()

axis_solution.set_xlabel("x")
axis_solution.set_ylabel("y")
axis_solution.set_xlim((0, 1))
axis_solution.set_ylim((0, 1))
color_bar = plt.colorbar(triangles_plot, ax=axis_solution)
color_bar.set_label(r"$H^1$ error")
axis_solution.set_title(r"interpolation to $H^1_0$")

fig_xd, axis_xd = plt.subplots()

tr_plot = PolyCollection(
    c4e,  # type: ignore
    array=xd.reshape(-1),
    cmap="viridis",
    edgecolors="black",
    linewidths=0.2,
)

axis_xd.add_collection(tr_plot)
axis_xd.autoscale_view()

axis_xd.set_xlabel("x")
axis_xd.set_ylabel("y")
axis_xd.set_xlim((0, 1))
axis_xd.set_ylim((0, 1))
color_bar = plt.colorbar(
    tr_plot,
    ax=axis_xd,
)
color_bar.set_label(r"$H^1$ error")
axis_xd.set_title(r"interpolation to $H^1$")

plt.show()
