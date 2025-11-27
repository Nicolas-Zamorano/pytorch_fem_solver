import abc
from typing import Callable, Optional, Tuple
import tqdm
import torch
from ..problems import AbstractProblem
from .abstract_solver import AbstractSolver
from ..mesh import AbstractMesh
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from ..model import FeedForwardNeuralNetwork as NeuralNetwork


class DeepSolver(AbstractSolver):
    """Deep Learning-based FEM solver."""

    def __init__(
        self,
        mesh: AbstractMesh,
        p_order: int,
        q_order: int,
        problem: AbstractProblem,
        neural_network: NeuralNetwork,
        error_mesh: Optional[AbstractMesh] = None,
        jit_compile: bool = True,
        posteriori_error: bool = False,
        epochs: int = 5000,
        optimizer: type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[dict] = None,
        learning_rate_scheduler: Optional[
            type[torch.optim.lr_scheduler.LRScheduler]
        ] = None,
        scheduler_kwargs: Optional[dict] = None,
        use_early_stopping: bool = False,
        early_stopping_patience: int = 10,
        min_delta: float = 1e-12,
        second_optimizer: Optional[type[torch.optim.Optimizer]] = torch.optim.LBFGS,
        second_optimizer_kwargs: Optional[dict] = None,
        second_optimizer_epochs: Optional[int] = None,
    ):
        if jit_compile:
            self._neural_network = torch.jit.script(neural_network)
        else:
            self._neural_network = neural_network

        self._posteriori_error = posteriori_error

        self._epochs = epochs

        if optimizer_kwargs is None:
            optimizer_kwargs = {"lr": 0.001}
        self._optimizer = optimizer(
            self._neural_network.parameters(), **optimizer_kwargs
        )
        if isinstance(self._optimizer, torch.optim.LBFGS):
            self._closure = self.define_closure()

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

        self._loss_history: list[float] = []
        self._validation_loss_history: list[float] = []
        self._accuracy_history: list[float] = []

        self._progress_bar = tqdm.tqdm(range(self._epochs), desc="Training Progress")

        self._best_loss = float("inf")
        self.optimal_parameters = self._neural_network.state_dict()

        if self._use_early_stopping:
            self.early_stopping_counter = 0

        if second_optimizer is not None:
            if second_optimizer_kwargs is None:
                second_optimizer_kwargs = optimizer_kwargs
            self.second_optimizer = second_optimizer(
                self._neural_network.parameters(),
                **second_optimizer_kwargs,
            )
            self.second_optimizer_epochs = second_optimizer_epochs
            if isinstance(self.second_optimizer, torch.optim.LBFGS):
                self._closure = self.define_closure()

        super().__init__(mesh, p_order, q_order, problem, error_mesh)

    def _training_step(
        self, neural_network: NeuralNetwork | torch.jit.ScriptModule
    ) -> Tuple[torch.Tensor, ...]:
        """Perform a single training step."""

        loss_value = self._compute_loss(neural_network)

        _, h1_error = self.compute_error(loss_value)

        relative_loss = (
            torch.sqrt(loss_value) / self.precomputed_values["exact_H1_norm"]
        )

        return (
            loss_value,
            relative_loss,
            h1_error / self.precomputed_values["exact_H1_norm"],
        )

    def solve(self):
        """Train the neural network."""
        for epochs in self._progress_bar:
            if isinstance(self._optimizer, torch.optim.LBFGS):
                self._optimizer.step(self._closure)
                loss_value_float = self._loss_history[-1]
                relative_loss_float = self._validation_loss_history[-1]
                accuracy_float = self._accuracy_history[-1]
            else:
                self._optimizer.zero_grad()
                loss, validation_loss, accuracy = self._training_step(
                    self._neural_network
                )
                loss.backward()
                self._optimizer.step()

                loss_value_float = loss.item()
                relative_loss_float = validation_loss.item()
                accuracy_float = accuracy.item()

                self._loss_history.append(loss_value_float)
                self._validation_loss_history.append(relative_loss_float)
                self._accuracy_history.append(accuracy_float)

            if self._learning_rate_scheduler is not None:
                self._learning_rate_scheduler.step()

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
                    "Loss": f"{loss_value_float:.8e}",
                    "Validation loss": f"{relative_loss_float:.8e}",
                    "Accuracy": f"{accuracy_float:.8e}",
                }
            )

            if epochs == self.second_optimizer_epochs:
                self._optimizer = self.second_optimizer

        self._neural_network.load_state_dict(self.optimal_parameters)

        nn_value_dofs = (
            self._neural_network(
                self.basis.coordinates_4_global_dofs.unsqueeze(-2).unsqueeze(-2)
            )
            .squeeze(-2)
            .squeeze(-2)
        )

        return nn_value_dofs

    def get_training_history(self):
        """Get the history of training losses."""
        return self._loss_history, self._validation_loss_history, self._accuracy_history

    def define_closure(self) -> Callable[[], float]:
        """Define closure for optimizers like LBFGS."""

        def closure() -> float:
            self._optimizer.zero_grad()
            loss, validation_loss, accuracy = self._training_step(self._neural_network)
            loss.backward()
            loss_value_float = loss.item()
            self._loss_history.append(loss_value_float)
            self._validation_loss_history.append(validation_loss.item())
            self._accuracy_history.append(accuracy.item())
            return loss_value_float

        return closure

    @abc.abstractmethod
    def _compute_loss(
        self,
        neural_network: NeuralNetwork | torch.jit.ScriptModule,
    ) -> torch.Tensor:
        """Computes the Loss."""
        raise NotImplementedError

    def compute_error(self, numerical_solution):

        neural_network_value, neural_network_grad = (
            self._neural_network.value_and_gradient(self.basis.integration_points)
        )

        neural_network_dx, neural_network_dy = torch.split(neural_network_grad, 1, -1)

        L2_error = self.basis.integrate_functional(
            self.problem.precomputed_L2_norm,
            neural_network_value - self.precomputed_values["exact_value"],  # type: ignore
        )

        H1_error = self.basis.integrate_functional(
            self.problem.precomputed_H1_norm,
            neural_network_value - self.precomputed_values["exact_value"],  # type: ignore
            neural_network_dx - self.precomputed_values["exact_dx_value"],
            neural_network_dy - self.precomputed_values["exact_dy_value"],
        )

        return L2_error, H1_error

    def plot(self, numerical_solution: torch.Tensor) -> Tuple[Figure, Figure, Figure]:
        L2_error, H1_error = self._plot(numerical_solution)
        loss_history, validation_loss_history, accuracy_history = (
            self.get_training_history()
        )

        figure_training, (axis_history, axis_robustness) = plt.subplots(
            1, 2, figsize=(10, 4)
        )

        axis_history.plot(loss_history, "-", label=r"$\mathcal{L}(u_{\theta})$")
        axis_history.plot(
            validation_loss_history,
            "--",
            label=r"$\frac{\sqrt{\mathcal{L}(u_{\theta})}}{\|u\|_U}$",
        )
        axis_history.plot(
            accuracy_history,
            ":",
            label=r"$\frac{\|u-u_{\theta}\|_U}{\|u\|_U}$",
        )
        axis_history.legend()
        axis_history.set_yscale("log")
        axis_history.set_xlabel("Epochs")
        axis_history.set_ylabel("Value")
        axis_history.set_title("Training History")
        axis_history.grid(True)

        robustness = [
            valiation_loss / accuracy_history
            for valiation_loss, accuracy_history in zip(
                validation_loss_history, accuracy_history
            )
        ]

        axis_robustness.plot(
            robustness,
            ":",
            label=r"$\frac{\sqrt{\mathcal{L}(u_{\theta})}}{\|u-u_{\theta}\|_U}$",
        )
        axis_robustness.set_xlabel("Epochs")
        axis_robustness.set_ylabel("Value")
        axis_robustness.set_title("Training Robustness")
        axis_robustness.legend()
        axis_robustness.grid(True)
        figure_training.tight_layout()

        return L2_error, H1_error, figure_training
