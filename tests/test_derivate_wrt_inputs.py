"Test derivate of NN w.r.t inputs." ""

import torch
import triangle as tr
from tensordict import TensorDict

from fem import Basis, ElementTri, MeshTri
from model import FeedForwardNeuralNetwork as NeuralNetwork


# def test_derivate_wrt_inputs():
#     """Test the derivative of the neural network with respect to its inputs."""
torch.set_default_dtype(torch.float64)

neural_network = NeuralNetwork(
    input_dimension=2,
    output_dimension=1,
    nb_hidden_layers=4,
    neurons_per_layers=25,
    boundary_condition_modifier=lambda x: 1,
)

mesh_data = tr.triangulate(
    {"vertices": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]},
    "Dqena" + str(0.5**8),
)

mesh = MeshTri(triangulation=TensorDict(mesh_data))

elements = ElementTri(polynomial_order=1, integration_order=2)

basis = Basis(mesh, elements)

integration_points = basis.integration_points

x, y = torch.split(integration_points, 1, dim=-1)

gradients = neural_network.gradient(integration_points)

STEP_SIZE = 2**-9

dx_finite_difference = (
    neural_network(torch.cat([x + STEP_SIZE, y], dim=-1))
    - neural_network(torch.cat([x - STEP_SIZE, y], dim=-1))
) / (2 * STEP_SIZE)

dy_finite_difference = (
    neural_network(torch.cat([x, y + STEP_SIZE], dim=-1))
    - neural_network(torch.cat([x, y - STEP_SIZE], dim=-1))
) / (2 * STEP_SIZE)

dx_automatic_differentiation, dy_automatic_differentiation = torch.split(
    gradients, 1, dim=-1
)

assert torch.allclose(dx_finite_difference, dx_automatic_differentiation)
assert torch.allclose(dy_finite_difference, dy_automatic_differentiation)

laplacian_automatic_differentiation = neural_network.laplacian(integration_points)

d2x_finite_difference = (
    neural_network(torch.cat([x + STEP_SIZE, y], dim=-1))
    - 2 * neural_network(integration_points)
    + neural_network(torch.cat([x - STEP_SIZE, y], dim=-1))
) / (STEP_SIZE**2)

d2y_finite_difference = (
    neural_network(torch.cat([x, y + STEP_SIZE], dim=-1))
    - 2 * neural_network(integration_points)
    + neural_network(torch.cat([x, y - STEP_SIZE], dim=-1))
) / (STEP_SIZE**2)

laplacian_finite_difference = d2x_finite_difference + d2y_finite_difference

assert torch.allclose(laplacian_finite_difference, laplacian_automatic_differentiation)
