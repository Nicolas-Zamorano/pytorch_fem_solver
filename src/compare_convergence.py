"""Compare convergence rates of VPINNs and FEM in H1 norm."""

import pickle
import numpy as np
import matplotlib.pyplot as plt

with open("H1_norm_converge_NN.pkl", "rb") as f:
    dofs_NN, errors_NN = pickle.load(f)

with open("H1_norm_converge_FEM.pkl", "rb") as f:
    dofs_FEM, errors_FEM = pickle.load(f)

dofs_NN = np.array(dofs_NN)
errors_NN = np.array(errors_NN)
dofs_FEM = np.array(dofs_FEM)
errors_FEM = np.array(errors_FEM)

log_dofs_NN = np.log10(dofs_NN)
log_errors_NN = np.log10(errors_NN)
slope_NN, intercept_NN = np.polyfit(log_dofs_NN, log_errors_NN, 1)
fit_NN = 10**intercept_NN * dofs_NN**slope_NN

log_dofs_FEM = np.log10(dofs_FEM)
log_errors_FEM = np.log10(errors_FEM)
slope_FEM, intercept_FEM = np.polyfit(log_dofs_FEM, log_errors_FEM, 1)
fit_FEM = 10**intercept_FEM * dofs_FEM**slope_FEM

fig, ax = plt.subplots(dpi=500)
ax.loglog(
    dofs_FEM,
    errors_FEM,
    "s",
    color="blue",
    markersize=7,
    markeredgecolor="black",
    label=f"FEM (decay = {-slope_FEM:.2f})",
)
ax.loglog(dofs_FEM, fit_FEM, ":", color="blue", alpha=0.5)

ax.loglog(
    dofs_NN,
    errors_NN,
    "^",
    color="orange",
    markersize=7,
    markeredgecolor="black",
    label=f"VPINNs (decay rate = {-slope_NN:.2f})",
)

ax.loglog(dofs_NN, fit_NN, "-.", color="orange", alpha=0.5)
ax.set_xlabel("# DOFs")
ax.set_ylabel("Relative $H^1$ Error")
ax.legend()
ax.grid(True, which="both", linestyle="--", alpha=0.3)

# plt.title("Convergence comparison: NN vs FEM")
plt.tight_layout()
plt.show()
