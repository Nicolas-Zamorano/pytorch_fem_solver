"""Module for neural networks"""

from typing import List, Optional, Tuple
import torch


class IdentityBC(torch.nn.Module):
    """base class for strong application of  boundary conditions"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward pass"""
        return torch.ones_like(x[..., :1])


class DistanceFunctionBC(torch.nn.Module):
    """Distance layer for strong application of boundary conditions"""

    def __init__(self, segments_points: torch.Tensor):
        super().__init__()
        self.segments_points = segments_points.unsqueeze(-3).unsqueeze(-3).unsqueeze(-3)
        self.normalization_order = 1.0

        self.diff_points, self.length, self.segments_midpoint = (
            self.compute_segment_values(self.segments_points)
        )

    def compute_segment_values(
        self, segments_points: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute values used in the distance function."""
        diff_points = segments_points[..., [1], :] - segments_points[..., [0], :]
        length = torch.norm(diff_points, dim=-1, keepdim=True)
        segments_midpoint = segments_points.mean(dim=-2, keepdim=True)

        return diff_points, length, segments_midpoint

    def linseg(self, points: torch.Tensor) -> torch.Tensor:
        """Compute the Rvachev function for a set of line segments."""
        diff_segments_points = points - self.segments_points[..., [0], :]

        signed_distance_function = (1 / self.length) * (
            diff_segments_points[..., [0]] * self.diff_points[..., [1]]
            - diff_segments_points[..., [1]] * self.diff_points[..., [0]]
        )

        trimming_function = (1.0 / self.length) * (
            (self.length / 2.0) ** 2
            - torch.norm(points - self.segments_midpoint, dim=-1, keepdim=True) ** 2
        )

        varphi = torch.sqrt(trimming_function**2 + signed_distance_function**4)

        phi = torch.sqrt(
            signed_distance_function**2 + 0.25 * (varphi - trimming_function) ** 2
        )
        return phi

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """Normalized Rvachev function for a set of line segments."""
        phi_val = self.linseg(points)
        rvachev_function = 1.0 / torch.sqrt(
            (1.0 / (phi_val**self.normalization_order)).sum(0)
        )
        return rvachev_function.squeeze(0)


class FeedForwardNeuralNetwork(torch.nn.Module):
    """Feed-Forward Neural Network (FNN) class compatible with TorchScript."""

    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        nb_hidden_layers: int,
        neurons_per_layers: int,
        activation_function: torch.nn.Module = torch.nn.Tanh(),
        use_xavier_initialization: bool = False,
        boundary_condition_modifier: Optional[torch.nn.Module] = None,
        boundary_condition_value: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self._input_dimension = input_dimension
        self._output_dimension = output_dimension
        self._nb_hidden_layers = nb_hidden_layers
        self._neurons_per_layers = neurons_per_layers
        self._activation_function = activation_function
        self._use_xavier_initialization = use_xavier_initialization

        if boundary_condition_modifier is None:
            self._boundary_condition_modifier = IdentityBC()
        else:
            self._boundary_condition_modifier = boundary_condition_modifier

        if boundary_condition_value is None:
            self._boundary_condition_value = torch.zeros(1)
        else:
            self._boundary_condition_value = boundary_condition_value

        self._neural_network = self.build_network(
            input_dimension,
            output_dimension,
            nb_hidden_layers,
            neurons_per_layers,
            activation_function,
            use_xavier_initialization,
        )

    def build_network(
        self,
        input_dimension: int,
        output_dimension: int,
        nb_layers: int,
        neurons_per_layers: int,
        activation_function: torch.nn.Module,
        use_xavier_initialization: bool,
    ) -> torch.nn.Sequential:
        """Build the neural network architecture."""
        layers = []

        layers.append(torch.nn.Linear(input_dimension, neurons_per_layers))
        layers.append(activation_function)

        for _ in range(nb_layers):
            layers.append(torch.nn.Linear(neurons_per_layers, neurons_per_layers))
            layers.append(activation_function)

        layers.append(torch.nn.Linear(neurons_per_layers, output_dimension))

        seq = torch.nn.Sequential(*layers)

        if use_xavier_initialization:
            for layer in seq:
                if isinstance(layer, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(layer.weight)

        return seq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return (
            self._neural_network(x) * self._boundary_condition_modifier(x)
        ) + self._boundary_condition_value

    @torch.jit.export
    def value_and_gradient(
        self, inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the gradient of the neural network with respect to its inputs."""
        inputs.requires_grad_(True)
        output = self.forward(inputs)

        grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(output)]

        gradients = torch.autograd.grad(
            outputs=[output],
            inputs=[inputs],
            grad_outputs=grad_outputs,  # type: ignore
            retain_graph=True,
            create_graph=True,
        )[0]

        assert gradients is not None

        return output, gradients

    @torch.jit.export
    def value_and_laplacian(
        self, inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the laplacian of the neural network with respect to its inputs."""
        inputs.requires_grad_(True)
        output = self.forward(inputs)

        grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(output)]

        gradients: Optional[torch.Tensor] = torch.autograd.grad(
            outputs=[output],
            inputs=[inputs],
            grad_outputs=grad_outputs,  # type: ignore
            retain_graph=True,
            create_graph=True,
        )[0]

        assert gradients is not None

        laplacian = torch.zeros_like(output)

        for i in range(inputs.shape[-1]):

            gradient: torch.Tensor = gradients[..., i]
            gradient_outputs: List[Optional[torch.Tensor]] = [
                torch.ones_like(output).squeeze(-1)
            ]
            grad2 = torch.autograd.grad(
                [gradient],
                [inputs],
                grad_outputs=gradient_outputs,  # type: ignore
                create_graph=True,
                retain_graph=True,
            )[0]
            assert grad2 is not None
            laplacian += grad2[..., i : i + 1]

        return output, gradients, laplacian
