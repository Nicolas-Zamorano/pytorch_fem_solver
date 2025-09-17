"""Compare convergence rates of VPINNs and FEM in H1 norm."""

import pickle
import numpy as np
import matplotlib.pyplot as plt

with open("H1_norm_converge_NN.pkl", "rb") as f:
    dofs_neural_network, errors_neural_network = pickle.load(f)

with open("H1_norm_converge_FEM.pkl", "rb") as f:
    dofs_finite_element, errors_finite_element = pickle.load(f)

dofs_neural_network = np.array(dofs_neural_network)
errors_neural_network = np.array(errors_neural_network)
dofs_finite_element = np.array(dofs_finite_element)
errors_finite_element = np.array(errors_finite_element)

log_dofs_neural_network = np.log10(dofs_neural_network)
log_errors_neural_network = np.log10(errors_neural_network)
slope_neural_network, intercept_neural_network = np.polyfit(
    log_dofs_neural_network, log_errors_neural_network, 1
)
fit_NN = 10**intercept_neural_network * dofs_neural_network**slope_neural_network

log_dofs_finite_element = np.log10(dofs_finite_element)
log_errors_finite_element = np.log10(errors_finite_element)
slope_finite_element, intercept_finite_element = np.polyfit(
    log_dofs_finite_element, log_errors_finite_element, 1
)
fit_finite_element = (
    10**intercept_finite_element * dofs_finite_element**slope_finite_element
)

fig, ax = plt.subplots(dpi=500)
ax.loglog(
    dofs_finite_element,
    errors_finite_element,
    "s",
    color="blue",
    markersize=7,
    markeredgecolor="black",
    label=f"FEM (decay = {-slope_finite_element:.2f})",
)
ax.loglog(dofs_finite_element, fit_finite_element, ":", color="blue", alpha=0.5)

ax.loglog(
    dofs_neural_network,
    errors_neural_network,
    "^",
    color="orange",
    markersize=7,
    markeredgecolor="black",
    label=f"VPINNs (decay rate = {-slope_neural_network:.2f})",
)

ax.loglog(dofs_neural_network, fit_NN, "-.", color="orange", alpha=0.5)
ax.set_xlabel("# DOFs")
ax.set_ylabel("Relative $H^1$ Error")
ax.legend()
ax.grid(True, which="both", linestyle="--", alpha=0.3)

# plt.title("Convergence comparison: NN vs FEM")
plt.tight_layout()
plt.show()
