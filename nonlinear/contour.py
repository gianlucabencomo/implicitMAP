import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from typing import Tuple, Any

from data import Nonlinear, Lorenz, GRW
from optimizers import Adam
from filter import MAP, KalmanFilter

plt.style.use("seaborn-v0_8-paper")


def init_data(seed: int, system: str) -> Tuple[Any, bool, int, int]:
    """Returns data class, linearity status, and state / obs space dim."""
    np.random.seed(seed)
    if system == "lorenz":
        n, m = 3, 3
        data, nonlinear = Lorenz(), True
    elif system == "nonlinear":
        n, m = 1, 1
        data, nonlinear = Nonlinear(), True
    elif system == "grw":
        n, m = 1, 1
        data, nonlinear = GRW(n=n), False
    else:
        raise ValueError("Unrecognized System.")
    return data, nonlinear, n, m


def objective(x: np.array) -> np.array:
    """Gaussian likelihood used in system 1 and 2."""
    z = np.zeros_like(x)
    for i in range(z.shape[0]):
        v = y[k] - h(x[i], k)
        z[i] = v.T @ inv(R(x[i], k)) @ v
    return z


def gradient(x: np.array) -> np.array:
    """First derivative of Gaussian likelihood."""
    g = np.zeros_like(x)
    for i in range(g.shape[0]):
        v = y[k] - h(x[i], k)
        g[i] = -H(x[i], k).T @ v
    return g


T = 35  # number of timesteps to plot
seed = 0
system = "nonlinear"
data, nonlinear, n, m = init_data(seed, system=system)
dt = 0.1
x_, y = data.forward(T)
h, H, f, F, R, Q = data.h, data.H, data.f, data.F, data.R, data.Q

# perform optimization and get steps for MAP
np.random.seed(seed)
optimizer = Adam(k=25, alpha=0.1, beta1=0.9, beta2=0.9)
map = MAP(n, m, optimizer)
map.initialize(data.H, data.f, data.h, dt=dt)
preds, steps = map.forward(y, True)

# EKF
np.random.seed(seed)
kf = KalmanFilter(n, m)
kf.initialize(data.F, data.H, data.Q, data.R, data.f, data.h, dt=dt)
preds_ekf, P = kf.forward(y)
preds_ekf = np.concatenate(
    [np.array([steps[0][0]]), preds_ekf], axis=0
)  # add init value to EKF

x = np.linspace(-30, 30, 1000).reshape(-1, 1)
# Loop through time steps
for k in range(T):
    np.random.seed(0)

    # Create subplots
    fig, axs = plt.subplots(2, 1, figsize=(8, 6))

    # Plot objective function
    axs[0].axhline(y=0, color="k", alpha=0.3)
    axs[0].axvline(x=x_[k], label=r"True State $x_t$", linewidth=2.5)
    axs[0].axvline(
        x=np.sqrt(y[k] * 20), c="r", label=r"Observation $h^{-1}(y_t)$", linewidth=2.5
    )
    axs[0].plot(x, objective(x), label="Likelihood", linewidth=2.5)
    axs[0].plot(
        steps[k], objective(steps[k]), "-o", label="Adam Update Steps", linewidth=2.5
    )
    axs[0].plot(
        preds_ekf[k : k + 2],
        objective(preds_ekf[k : k + 2]),
        "-o",
        label="EKF Update Step",
        linewidth=2.5,
    )
    axs[0].set_ylabel("Likelihood Value", fontsize=13, fontweight="bold")
    axs[0].set_title(f"Likelihood at T={k}", fontsize=13, fontweight="bold")
    axs[0].legend(
        fontsize=13,
    )
    axs[0].grid(True)
    axs[0].tick_params(axis="x", labelsize=13)  # Set x-axis tick size
    axs[0].tick_params(axis="y", labelsize=13)  # Set y-axis tick size
    axs[0].set_xlim(-30, 30)

    # Plot gradient
    axs[1].axhline(y=0, color="k", alpha=0.3)
    axs[1].axvline(x=x_[k], label=r"True State $x_t$", linewidth=2.5)
    axs[1].axvline(
        x=np.sqrt(y[k] * 20), c="r", label=r"Observation $h^{-1}(y_t)$", linewidth=2.5
    )
    axs[1].plot(
        x, gradient(x), label="Gradient of Likelihood", linestyle="--", linewidth=2.5
    )
    axs[1].set_xlabel("Domain", fontsize=13, fontweight="bold")
    axs[1].set_ylabel("Gradient Value", fontsize=13, fontweight="bold")
    axs[1].set_title(f"Gradient Visualization at T={k}", fontsize=13, fontweight="bold")
    axs[1].legend(
        fontsize=13,
    )
    axs[1].grid(True)
    axs[1].tick_params(axis="x", labelsize=13)  # Set x-axis tick size
    axs[1].tick_params(axis="y", labelsize=13)  # Set y-axis tick size
    axs[1].set_xlim(-30, 30)

    plt.tight_layout()
    plt.show()
