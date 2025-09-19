"""Module for neural networks"""

from typing import List, Optional
import torch


class IdentityBC(torch.nn.Module):
    """base class for strong application of  boundary conditions"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward pass"""
        return torch.ones_like(x[..., :1])


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
    def gradient(self, inputs: torch.Tensor) -> Optional[torch.Tensor]:
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

        return gradients

    # @torch.jit.export
    # def laplacian(self, inputs: torch.Tensor) -> torch.Tensor:
    #     """Compute the laplacian of the neural network with respect to its inputs."""
    #     inputs.requires_grad_(True)
    #     output = self.forward(inputs)

    #     grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(output)]

    #     gradients = torch.autograd.grad(
    #         outputs=[output],
    #         inputs=[inputs],
    #         grad_outputs=grad_outputs,
    #         retain_graph=True,
    #         create_graph=True,
    #     )[0]

    #     laplacian = torch.zeros_like(output)

    #     for i in range(inputs.size(-1)):
    #         gradient = gradients.index_select(-1, torch.tensor([i]))
    #         grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(gradient)]
    #         grad2 = torch.autograd.grad(
    #             [gradient],
    #             [inputs],
    #             grad_outputs=grad_outputs,
    #             create_graph=True,
    #             retain_graph=True,
    #         )[0][..., i : i + 1]
    #         laplacian += grad2

    #     return laplacian
