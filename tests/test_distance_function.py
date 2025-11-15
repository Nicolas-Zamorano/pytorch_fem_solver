"""Example of using Rvachev functions to define a boundary function for a square domain."""

import torch
import matplotlib.pyplot as plt
from torch_fem import DistanceFunctionBC


# def linseg(points: torch.Tensor, segments_points: torch.Tensor) -> torch.Tensor:
#     """Compute the Rvachev function for a set of line segments."""
#     segments_points = segments_points.unsqueeze(-3)
#     points = points.unsqueeze(-3)

#     segments_vector = segments_points[..., [1], :] - segments_points[..., [0], :]
#     length = torch.norm(segments_vector, dim=-1, keepdim=True)
#     segments_midpoint = segments_points.mean(dim=-2, keepdim=True)

#     diff_segments_points = points - segments_points[..., [0], :]

#     signed_distance_function = (1 / length) * (
#         diff_segments_points[..., [0]] * segments_vector[..., [1]]
#         - diff_segments_points[..., [1]] * segments_vector[..., [0]]
#     )

#     trimming_function = (1.0 / length) * (
#         (length / 2.0) ** 2
#         - torch.norm(points - segments_midpoint, dim=-1, keepdim=True) ** 2
#     )

#     varphi = torch.sqrt(trimming_function**2 + signed_distance_function**4)

#     phi = torch.sqrt(
#         signed_distance_function**2 + 0.25 * (varphi - trimming_function) ** 2
#     )
#     return phi


# def boundary_constrain(
#     points: torch.Tensor, segments_points: torch.Tensor
# ) -> torch.Tensor:
#     """Normalized Rvachev function for a set of line segments."""
#     normalization_order = 1.0
#     phi_val = linseg(points, segments_points)
#     rvachev_function = 1.0 / torch.sqrt((1.0 / (phi_val**normalization_order)).sum(-4))
#     return rvachev_function


segments = torch.tensor(
    [
        [[0.0, 0.0], [1.0, 0.0]],
        [[1.0, 0.0], [1.0, 1.0]],
        [[1.0, 1.0], [0.0, 1.0]],
        [[0.0, 1.0], [0.0, 0.0]],
    ]
)

boundary_constrain = DistanceFunctionBC(segments)

N = 200
plot_points = torch.linspace(-0.5, 1.5, N)
X, Y = torch.meshgrid(plot_points, plot_points, indexing="xy")
stack_points = (
    torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1).unsqueeze(-2).unsqueeze(-2)
)
phi_vals = boundary_constrain(stack_points).reshape(N, N)

plt.figure(figsize=(6, 5))
plt.contourf(X.numpy(), Y.numpy(), phi_vals.numpy(), levels=50, cmap="viridis")
plt.colorbar(label=r"$\phi(x, y)$ (distancia aprox. a polígono)")

# for s in segments:
#     plt.plot([s[0, 0], s[1, 0]], [s[0, 1], s[1, 1]], "r-", lw=2)

plt.gca().set_aspect("equal")
plt.title("Campo φ(x, y)")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
