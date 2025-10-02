"Test derivate of NN w.r.t inputs." ""

from typing import cast
import torch
import triangle as tr

from torch_fem import (
    Basis,
    ElementTri,
    MeshTri,
    FeedForwardNeuralNetwork as NeuralNetwork,
)

torch.set_default_dtype(torch.float32)


def test_derivate_wrt_inputs():
    """Test the derivative of the neural network with respect to its inputs."""

    if torch.get_default_dtype() == torch.float32:
        absolute_tolerance = 1e-4
    elif torch.get_default_dtype() == torch.float64:
        absolute_tolerance = 1e-8
    else:
        raise ValueError("Unsupported torch dtype")

    class BoundaryConstrain(torch.nn.Module):
        """Class to strongly apply bc"""

        def forward(self, inputs):
            """Boundary condition modifier function."""
            inputs_x, inputs_y = torch.split(inputs, 1, dim=-1)
            return inputs_x * (inputs_x - 1) * inputs_y * (inputs_y - 1)

    neural_network = torch.jit.script(
        NeuralNetwork(
            input_dimension=2,
            output_dimension=1,
            nb_hidden_layers=4,
            neurons_per_layers=25,
            boundary_condition_modifier=BoundaryConstrain(),
        )
    )

    mesh_data = tr.triangulate(
        {"vertices": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]},
        "Dqena" + str(0.5**8),
    )

    mesh = MeshTri(triangulation=mesh_data)

    elements = ElementTri(polynomial_order=1, integration_order=2)

    basis = Basis(mesh, elements)

    integration_points = basis.integration_points

    x, y = torch.split(integration_points, 1, dim=-1)

    gradients = cast(torch.Tensor, neural_network.gradient(integration_points))

    step_size = 2**-9

    dx_finite_difference = (
        neural_network(torch.cat([x + step_size, y], dim=-1))
        - neural_network(torch.cat([x - step_size, y], dim=-1))
    ) / (2 * step_size)

    dy_finite_difference = (
        neural_network(torch.cat([x, y + step_size], dim=-1))
        - neural_network(torch.cat([x, y - step_size], dim=-1))
    ) / (2 * step_size)

    dx_automatic_differentiation, dy_automatic_differentiation = torch.split(
        gradients, 1, dim=-1
    )

    assert torch.allclose(
        dx_finite_difference, dx_automatic_differentiation, atol=absolute_tolerance
    )
    assert torch.allclose(
        dy_finite_difference, dy_automatic_differentiation, atol=absolute_tolerance
    )

    laplacian_automatic_differentiation = neural_network.laplacian(integration_points)

    d2x_finite_difference = (
        neural_network(torch.cat([x + step_size, y], dim=-1))
        - 2 * neural_network(integration_points)
        + neural_network(torch.cat([x - step_size, y], dim=-1))
    ) / (step_size**2)

    d2y_finite_difference = (
        neural_network(torch.cat([x, y + step_size], dim=-1))
        - 2 * neural_network(integration_points)
        + neural_network(torch.cat([x, y - step_size], dim=-1))
    ) / (step_size**2)

    laplacian_finite_difference = d2x_finite_difference + d2y_finite_difference

    assert torch.allclose(
        laplacian_finite_difference,
        laplacian_automatic_differentiation,
        atol=absolute_tolerance * 1e2,
    )


if __name__ == "__main__":
    test_derivate_wrt_inputs()
