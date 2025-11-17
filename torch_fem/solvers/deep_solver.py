import abc
from typing import Callable, Optional, Tuple
import tqdm
import torch
from ..problems import AbstractProblem
from .abstract_solver import AbstractSolver
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection


class DeepSolver(AbstractSolver):
    """Deep Learning-based FEM solver."""

    def __init__(
        self,
        mesh,
        p_order: int,
        q_order: int,
        problem: AbstractProblem,
        neural_network: torch.nn.Module,
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

        self._loss_history = []
        self._validation_loss_history = []
        self._accuracy_history = []

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

        super().__init__(mesh, p_order, q_order, problem)

    def solve(self):
        """Train the neural network."""
        for epochs in self._progress_bar:
            if isinstance(self._optimizer, torch.optim.LBFGS):
                ### CHECK THIS PART ###
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

        return self.optimal_parameters

    def get_training_history(self):
        """Get the history of training losses."""
        return self._loss_history, self._validation_loss_history, self._accuracy_history

    def load_optimal_parameters(self):
        """Load the optimal parameters of the neural network."""
        self._neural_network.load_state_dict(self.optimal_parameters)


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
    def _training_step(
        self,
        neural_network: torch.nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Define a single training step."""
        raise NotImplementedError

    def compute_error(self, neural_network) -> Tuple[torch.Tensor, torch.Tensor]:

        neural_network_value, neural_network_grad = neural_network.value_and_gradient(
            self.basis.integration_points
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

    def plot(self, neural_network: NeuralNetwork):

        coordinates_4_triangles = self.mesh["cells", "coordinates"]
        coordinates_4_vertices = self.mesh["vertices", "coordinates"]

        exact_value = self.problem.exact(coordinates_4_vertices).squeeze(-1)
        numerical_solution = (
            neural_network(coordinates_4_vertices.unsqueeze(-2).unsqueeze(-2))
            .reshape(-1)
            .numpy(force=True)
        )

        L2_error, H1_error = self.compute_error(neural_network)

        x_min, x_max = (
            coordinates_4_vertices[:, 0].min().numpy(force=True),
            coordinates_4_vertices[:, 0].max().numpy(force=True),
        )
        y_min, y_max = (
            coordinates_4_vertices[:, 1].min().numpy(force=True),
            coordinates_4_vertices[:, 1].max().numpy(force=True),
        )
        z_exact_min, z_exact_max = exact_value.min().numpy(
            force=True
        ), exact_value.max().numpy(force=True)

        figure_solution, (axis_numerical_solution, axis_exact_solution) = plt.subplots(
            1, 2, figsize=(10, 4), subplot_kw={"projection": "3d"}
        )
        axis_numerical_solution.plot_trisurf(
            coordinates_4_vertices[:, 0].numpy(force=True),
            coordinates_4_vertices[:, 1].numpy(force=True),
            numerical_solution,
            triangles=coordinates_4_triangles,
            cmap="viridis",
            edgecolor="black",
            linewidth=0.2,
        )

        axis_numerical_solution.set_xlim(x_min, x_max)
        axis_numerical_solution.set_ylim(y_min, y_max)
        axis_numerical_solution.set_zlim(z_exact_min, z_exact_max)
        axis_numerical_solution.set_title("Numerical Solution")
        axis_numerical_solution.set_xlabel("x")
        axis_numerical_solution.set_ylabel("y")
        axis_numerical_solution.set_zlabel("u(x,y)")

        axis_exact_solution.plot_trisurf(
            coordinates_4_vertices[:, 0].numpy(force=True),
            coordinates_4_vertices[:, 1].numpy(force=True),
            exact_value.numpy(force=True),
            triangles=coordinates_4_triangles,
            cmap="viridis",
            edgecolor="black",
            linewidth=0.2,
        )

        axis_exact_solution.set_xlim(x_min, x_max)
        axis_exact_solution.set_ylim(y_min, y_max)
        axis_exact_solution.set_zlim(z_exact_min, z_exact_max)
        axis_exact_solution.set_title("Exact Solution")
        axis_exact_solution.set_xlabel("x")
        axis_exact_solution.set_ylabel("y")
        axis_exact_solution.set_zlabel("u(x,y)")

        figure_solution.tight_layout()

        figure_error, (axis_L2, axis_H1) = plt.subplots(1, 2, figsize=(10, 4))

        l2_error_surface = PolyCollection(
            coordinates_4_triangles,
            array=L2_error.sqrt().squeeze(-1).numpy(force=True),
            cmap="viridis",
            edgecolor="black",
            linewidths=0.2,
        )
        axis_L2.add_collection(l2_error_surface)
        axis_L2.set_xlim(x_min, x_max)
        axis_L2.set_ylim(y_min, y_max)
        axis_L2.set_aspect("equal")
        axis_L2.set_title(r"L2 Error = {:.4e}".format(L2_error.sum().sqrt().item()))
        figure_error.colorbar(l2_error_surface, ax=axis_L2)

        h1_error_surface = PolyCollection(
            coordinates_4_triangles,
            array=H1_error.sqrt().squeeze(-1).numpy(force=True),
            cmap="viridis",
            edgecolor="black",
            linewidths=0.2,
        )
        axis_H1.add_collection(h1_error_surface)
        axis_H1.set_xlim(x_min, x_max)
        axis_H1.set_ylim(y_min, y_max)
        axis_H1.set_aspect("equal")
        axis_H1.set_title(r"H1 Error = {:.4e}".format(H1_error.sum().sqrt().item()))
        figure_error.colorbar(h1_error_surface, ax=axis_H1)

        figure_error.tight_layout()

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

        return (
            figure_solution,
            figure_error,
            figure_training,
        )
