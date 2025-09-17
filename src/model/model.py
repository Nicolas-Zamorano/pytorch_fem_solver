"""Model base class for training"""

import matplotlib.pyplot as plt
import torch
import tqdm


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
