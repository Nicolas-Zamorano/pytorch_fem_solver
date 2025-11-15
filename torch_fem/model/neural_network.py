"""Module for neural networks"""

from typing import List, Optional, Tuple
import torch


class IdentityBC(torch.nn.Module):
    """base class for strong application of  boundary conditions"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward pass"""
        return torch.ones_like(x[..., :1])


class DistanceFunctionBC(torch.nn.Module):
    """Distance layer for strong application of boundary conditions.

    Uses an Rvachev-style distance function for a set of line segments.
    """

    def __init__(self, segments_points: torch.Tensor):
        super().__init__()
        self.segments_points = segments_points.unsqueeze(-3)
        self.segments_endpoints_first_point = torch.index_select(
            self.segments_points, -2, torch.tensor([0], dtype=torch.long)
        )
        self.normalization_power = 1.0

        (self.segment_vectors, self.segment_lengths, self.segment_midpoints) = (
            self.compute_segment_values(self.segments_points)
        )

        (
            self.segment_vectors_first_coordinate,
            self.segment_vectors_second_coordinate,
        ) = torch.split(self.segment_vectors, 1, dim=-1)

    def compute_segment_values(
        self, segments_points: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute per-segment vectors, lengths and midpoints."""
        segments_first_point, segments_second_point = torch.split(
            segments_points, 1, dim=-2
        )
        segment_vectors = segments_second_point - segments_first_point
        segment_lengths = torch.norm(segment_vectors, dim=-1, keepdim=True)
        segment_midpoints = segments_points.mean(dim=-2, keepdim=True)

        return (
            segment_vectors,
            segment_lengths,
            segment_midpoints,
        )

    def compute_rvachev_for_segments(self, points: torch.Tensor) -> torch.Tensor:
        """Compute the Rvachev distance-like value for each line segment.

        Returns a tensor of per-segment distance contributions for the input points.
        """
        points = points.unsqueeze(-4)

        vectors_from_segment_start = points - self.segments_endpoints_first_point

        (
            vectors_from_segment_start_first_coordinate,
            vectors_from_segment_start_second_coordinate,
        ) = torch.split(vectors_from_segment_start, 1, dim=-1)

        signed_distances = (1 / self.segment_lengths) * (
            vectors_from_segment_start_first_coordinate
            * self.segment_vectors_second_coordinate
            - vectors_from_segment_start_second_coordinate
            * self.segment_vectors_first_coordinate
        )

        trimming_values = (1.0 / self.segment_lengths) * (
            (self.segment_lengths / 2.0) ** 2
            - torch.norm(points - self.segment_midpoints, dim=-1, keepdim=True) ** 2
        )

        function_value = torch.sqrt(trimming_values**2 + signed_distances**4)

        segment_distance = torch.sqrt(
            signed_distances**2 + 0.25 * (function_value - trimming_values) ** 2
        )
        return segment_distance

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """Normalized Rvachev function aggregated over segments for the given points."""
        per_segment_distances = self.compute_rvachev_for_segments(points)
        normalized_rvachev = 1.0 / torch.sqrt(
            (1.0 / (per_segment_distances**self.normalization_power)).sum(-4)
        )
        return normalized_rvachev


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
        return self._neural_network(x) * self._boundary_condition_modifier(x)

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
