import typer
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Any

from data import GRW, Nonlinear, Lorenz
from filter import MAP, KalmanFilter, UnscentedKF, Bootstrap
from optimizers import Adam, Santos, RMSprop, GD

from helper import RMSE, plot1d, plothist

plt.style.use("seaborn-v0_8-paper")


def init_main(seed: int, system: str) -> Tuple[Any, bool, int, int]:
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


def main(
    seed: int = 0,
    T: int = 200,
    n_particles: int = 1000,
    system: str = "grw",
    optimizer: str = "Santos",
    plot: bool = False,
    verbose: bool = False,
) -> None:
    # initialize system, random seed
    data, nonlinear, n, m = init_main(seed, system)
    dt = 0.02 if system == "lorenz" else 0.1

    if system == "lorenz":
        x_init = np.random.normal(
            10, 1, size=(n, 1)
        )  # sample from isotropic gaussian prior centered at 10
    else:
        x_init = np.random.normal(
            0, 1, size=(n, 1)
        )  # sample from isotropic gaussian prior

    # init plotting
    if plot:
        if n == 1:
            _ = plt.figure(figsize=(7, 5))
            plt.ylim([-30, 30])
            plt.xlim([-5, T + 5])
        elif n == 3:
            ax = plt.figure(figsize=(7, 4.5)).add_subplot(projection="3d")

    # get data
    x_, y = data.forward(T)

    # -----------------------------------------------------------------
    # Particle filter baseline
    if system == "lorenz":
        x_pf_init = np.random.normal(
            10, 5, size=(n_particles, n, 1)
        )  # sample from isotropic gaussian prior centered at 10
    else:
        x_pf_init = None
    pf = Bootstrap(n, m, n_particles=n_particles)
    pf.initialize(data.Q, data.R, data.f, data.h, x=x_pf_init, dt=dt)
    start = time.time()
    x, _ = pf.forward(y)
    end = time.time()
    pf_mean = x.mean(axis=1).reshape(T, n)  # get mean
    rmse = RMSE(pf_mean, x_.reshape(T, n))
    pf_x = x

    if verbose:
        print(f"PF Baseline: RMSE = {rmse:.3f}, Runtime = {end - start:.3f}")

    if plot:
        if n == 1:
            plot1d(pf_mean, label=f"PF Mean", c="gray")
    # -----------------------------------------------------------------

    # -----------------------------------------------------------------
    # run Kalman filter / Extended Kalman Filter depending on linearity
    kf = KalmanFilter(n, m)
    kf.initialize(data.F, data.H, data.Q, data.R, data.f, data.h, x=x_init, dt=dt)
    if system == "grw":
        Phat = kf.riccati()
    start = time.time()
    x, P = kf.forward(y)
    end = time.time()

    rmse = RMSE(x, x_.squeeze())
    if verbose:
        print(f"EKF Baseline: RMSE = {rmse:.3f}, Runtime = {end - start:.3f}")

    if plot:
        if n == 1:
            plot1d(
                x,
                label=f"EKF" if nonlinear else f"Kalman Filter",
            )
    # -----------------------------------------------------------------

    # -----------------------------------------------------------------
    # run Unscented Kalman Filter
    ukf = UnscentedKF(n, m)
    ukf.initialize(data.F, data.H, data.Q, data.R, data.f, data.h, x=x_init, dt=dt)
    x, P = ukf.forward(y)

    rmse = RMSE(x, x_.squeeze())
    if verbose:
        print(f"UKF Baseline: RMSE = {rmse:.3f}, Runtime = {end - start:.3f}")

    if plot:
        if n == 1:
            plot1d(x, label=f"UKF")
    # -----------------------------------------------------------------

    # -----------------------------------------------------------------
    # MAP
    if optimizer == "Santos":
        optimizer = Santos(np.eye(n) if nonlinear else Phat, 1, nonlinear)
    elif optimizer == "Adam":
        optimizer = Adam(k=25, alpha=0.25, beta1=0.1, beta2=0.1)
    elif optimizer == "RMSprop":
        optimizer = RMSprop(k=75, alpha=0.075, gamma=0.1)
    else:
        raise ValueError("Unknown optimizer.")
    if system == "lorenz":
        optimizer = GD(k=3, alpha=0.05)
        map_init = np.random.normal(10, 5, size=(n, 1))
    else:
        map_init = None
    map = MAP(n, m, optimizer)
    map.initialize(data.H, data.f, data.h, x=map_init, dt=dt)
    start = time.time()
    x = map.forward(y)
    end = time.time()
    map_x = x

    rmse = RMSE(x, x_.squeeze())
    if verbose:
        print(f"MAP: RMSE = {rmse:.3f}, Runtime = {end - start:.3f}")

    if plot:
        if n == 1:
            plot1d(x, label=f"MAP")
        elif n == 3:
            ax.plot(*x.T, "-", lw=2.5, label=f"MAP")
    # -----------------------------------------------------------------

    if plot:
        if n == 1:
            plot1d(x_.squeeze(), label="True State")
            if system != "lorenz" and n_particles >= 10000:
                plothist(pf_x, map_x, x_true=x_.squeeze())
        else:
            ax.plot(*x_.squeeze().T, "ko", markersize=4.2, label="True State")
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
            ax.view_init(elev=30, azim=315)
            ax.legend(
                loc="upper center",
                fontsize=16,
                ncols=2,
                frameon=False,
                bbox_to_anchor=(0.5, 1.1),
            )

        # Set the DPI (e.g., 300)
        dpi = 600
        # Save the figure with the desired DPI
        plt.savefig("example.png", dpi=dpi)
        plt.show()


if __name__ == "__main__":
    typer.run(main)
