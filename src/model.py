"# Definition of a Deep Neural Network (DNN) model using PyTorch."

import torch
import tqdm

import matplotlib.pyplot as plt


class FeedForwardNeuralNetwork(torch.nn.Module):
    """Feed-Forward Neural Network (FNN) class."""

    def __init__(
        self,
        input_dimension,
        output_dimension,
        nb_hidden_layers,
        neurons_per_layers,
        activation_function=torch.nn.Tanh(),
        use_xavier_initialization=False,
        boundary_condition_modifier=lambda x: 1,
    ):
        super().__init__()
        self._input_dimension = input_dimension
        self._output_dimension = output_dimension
        self._nb_hidden_layers = nb_hidden_layers
        self._neurons_per_layers = neurons_per_layers
        self._activation_function = activation_function
        self._use_xavier_initialization = use_xavier_initialization
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
        input_dimension,
        output_dimension,
        nb_layers,
        neurons_per_layers,
        activation_function,
        use_xavier_initialization,
    ):
        """Build the neural network architecture."""
        layers = []

        layers.append(torch.nn.Linear(input_dimension, neurons_per_layers))
        layers.append(activation_function)

        for _ in range(nb_layers):
            layers.append(torch.nn.Linear(neurons_per_layers, neurons_per_layers))
            layers.append(activation_function)

        layers.append(torch.nn.Linear(neurons_per_layers, output_dimension))

        if use_xavier_initialization:
            for layer in layers:
                if isinstance(layer, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(layer.weight)

        return torch.nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass through the network."""
        return self._neural_network(x) * self._boundary_condition_modifier(x)

    def gradient(self, x):
        """Compute the gradient of the network output with respect to its inputs."""
        x.requires_grad_(True)
        output = self.forward(x)
        gradients = torch.autograd.grad(
            outputs=output,
            inputs=x,
            grad_outputs=torch.ones_like(output),
            retain_graph=True,
            create_graph=True,
        )[0]
        return gradients


class Model:
    """Model class for training a neural network."""

    def __init__(
        self,
        neural_network: torch.nn.Module,
        training_step: callable,
        epochs: int = 5000,
        optimizer: type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: dict = {"lr": 1e-3},
        learning_rate_scheduler: type[torch.optim.lr_scheduler.LRScheduler] = None,
        scheduler_kwargs: dict = {},
        use_early_stopping: bool = False,
        early_stopping_patience: int = 10,
        min_delta: float = 1e-12,
    ):
        self._neural_network = neural_network
        self._training_step = training_step
        self._epochs = epochs
        self._optimizer = optimizer(
            self._neural_network.parameters(), **optimizer_kwargs
        )

        if learning_rate_scheduler is not None:
            self._learning_rate_scheduler = learning_rate_scheduler(
                self._optimizer, **scheduler_kwargs
            )

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

    def plot_training_history(self):
        """Plot the training history."""

        _, axis_loss = plt.subplots()
        axis_loss.semilogy(self._loss_history, label="Training loss")
        axis_loss.semilogy(self._validation_loss_history, label="Validation loss")
        axis_loss.semilogy(self._accuracy_history, label="Accuracy")
        axis_loss.set_xlabel("# Epochs")
        axis_loss.set_ylabel("Loss")
        axis_loss.set_title("Training history")
        axis_loss.legend()
