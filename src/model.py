"# Definition of a Deep Neural Network (DNN) model using PyTorch."

from typing import List, Optional

import matplotlib.pyplot as plt
import torch
import tqdm


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
        boundary_condition_modifier: torch.nn.Module = None,
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
            grad_outputs=grad_outputs,
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


class Model:
    """Model class for training a neural network."""

    def __init__(
        self,
        neural_network: torch.nn.Module,
        training_step: callable,
        epochs: int = 5000,
        optimizer: type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: dict = None,
        learning_rate_scheduler: type[torch.optim.lr_scheduler.LRScheduler] = None,
        scheduler_kwargs: dict = None,
        use_early_stopping: bool = False,
        early_stopping_patience: int = 10,
        min_delta: float = 1e-12,
    ):
        self._neural_network = torch.jit.script(neural_network)
        self._training_step = training_step
        self._epochs = epochs
        if optimizer_kwargs is None:
            optimizer_kwargs = {"lr": 0.001}
        self._optimizer = optimizer(
            self._neural_network.parameters(), **optimizer_kwargs
        )
        if scheduler_kwargs is None:
            scheduler_kwargs = {}

        if learning_rate_scheduler is not None:
            self._learning_rate_scheduler = learning_rate_scheduler(
                self._optimizer, **scheduler_kwargs
            )
        else:
            self._learning_rate_scheduler = None

        self._use_early_stopping = use_early_stopping
        self._early_stopping_patience = early_stopping_patience
        self._min_delta = min_delta

        self._loss_history = []
        self._validation_loss_history = []
        self._accuracy_history = []

        self._progress_bar = tqdm.tqdm(range(self._epochs), desc="Training Progress")

        self._best_loss = float("inf")
        self.optimal_parameters = self._neural_network.state_dict()

        if self._use_early_stopping:
            self.early_stopping_counter = 0

    def train(self):
        """Train the neural network."""
        for _ in self._progress_bar:
            self._optimizer.zero_grad()
            loss, validation_loss, accuracy = self._training_step(self._neural_network)
            loss.backward()
            self._optimizer.step()
            if self._learning_rate_scheduler is not None:
                self._learning_rate_scheduler.step(loss.detach())

            loss_value_float = loss.item()
            relative_loss_float = validation_loss.item()
            accuracy_float = accuracy.item()

            if self._use_early_stopping:
                if loss_value_float < self._best_loss - self._min_delta:
                    self._best_loss = loss_value_float
                    self.early_stopping_counter = 0
                    self.optimal_parameters = self._neural_network.state_dict()
                else:
                    self.early_stopping_counter += 1
                    if self.early_stopping_counter >= self._early_stopping_patience:
                        break
            else:
                if loss_value_float < self._best_loss:
                    self._best_loss = loss_value_float
                    self.optimal_parameters = self._neural_network.state_dict()

            self._progress_bar.set_postfix(
                {
                    "Loss": f"{loss_value_float:.8f}",
                    "Validation loss": f"{relative_loss_float:.8f}",
                    "Accuracy": f"{accuracy_float:.8f}",
                }
            )

            self._loss_history.append(loss_value_float)
            self._validation_loss_history.append(relative_loss_float)
            self._accuracy_history.append(accuracy_float)

    def get_training_history(self):
        """Get the history of training losses."""
        return self._loss_history, self._validation_loss_history, self._accuracy_history

    def load_optimal_parameters(self):
        """Load the optimal parameters of the neural network."""
        self._neural_network.load_state_dict(self.optimal_parameters)

    def plot_training_history(
        self,
        plot_names: dict = None,
    ):
        """Plot the training history."""
        if plot_names is None:
            plot_names = {
                "loss": "Training loss",
                "validation": "Validation loss",
                "accuracy": "Accuracy",
                "title": "Training history",
            }

        _, axis_loss = plt.subplots()
        axis_loss.semilogy(self._loss_history, label=plot_names["loss"])
        axis_loss.semilogy(
            self._validation_loss_history, label=plot_names["validation"]
        )
        axis_loss.semilogy(self._accuracy_history, label=plot_names["accuracy"])
        axis_loss.set_xlabel("# Epochs")
        axis_loss.set_ylabel("Loss")
        axis_loss.set_title(plot_names["title"])
        axis_loss.legend()
