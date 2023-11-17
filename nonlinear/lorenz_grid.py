# Code to run grid search for baseline methods (lorenz)

import typer
import numpy as np

from data import Lorenz, Nonlinear
from filter import KalmanFilter, MAP, IteratedEKF, UnscentedKF, Bootstrap
from helper import RMSE


def init_data(system: str, misspecified: bool):
    """Returns data class and state / obs space dim."""
    if system == "lorenz":
        n, m = 3, 3
        data = Lorenz(misspecified=misspecified)
    elif system == "nonlinear":
        n, m = 1, 1
        data = Nonlinear(misspecified=misspecified)
    else:
        raise ValueError("Unrecognized System.")
    return data, n, m


def init_dict(Q: np.array
):
    """Set up dictionary for running experiments."""
    res = {}
    for i in range(len(Q)):
        res["EKF" + str(np.diag(Q[i]).flatten())] = []
        res["UKF" + str(np.diag(Q[i]).flatten())] = []
    return res

def getQ():
    """Returns a list of Q's to test."""
    q = np.arange(0.01, 5.01, 0.01)
    Q_ = []
    for i in q:
        Q_.append(i * np.eye(3))
    return Q_

def main(
    N: int = 100,
    seed: int = 0,
    T: int = 200,
    verbose: bool = False,
    system: str = "lorenz",
    misspecified: bool = False,
):

    # init data + other configs
    data, n, m = init_data(system, misspecified)
    dt = 0.02 if system == "lorenz" else 0.1

    Q = getQ()

    # init dict
    res = init_dict(Q)

    if verbose:
        print(
            f"Testing {len(res)} models and baselines for {N} random experiments..."
        )

    # start experiments
    for i in range(N):  # experimental repititions
        if verbose:
            if i % 5 == 0 and i != 0:
                print(f"Completed [{i} / {N}].")

        np.random.seed(seed + i)
        # generate data based on seed
        x_, y = data.forward(T)
        if system == "lorenz":
            x_init = np.random.normal(
                10, 1, size=(n, 1)
            )  # sample from isotropic gaussian prior centered at 10
        else:
            x_init = np.random.normal(
                0, 1, size=(n, 1)
            )  # sample from isotropic gaussian prior centered at 0

        for q in Q:
            # ! Run Kalman Filter baseline
            kf = KalmanFilter(n, m)
            kf.initialize(
                data.F, data.H, lambda x, t: q, data.R, data.f, data.h, x=x_init, dt=dt
            )  # P init is np.eye(3)
            np.random.seed(seed + i)
            x, P = kf.forward(y)
            # append result
            res["EKF" + str(np.diag(q).flatten())].append(RMSE(x, x_.squeeze()))

            # ! Run Unscented Kalman Filter baseline
            ukf = UnscentedKF(n, m)
            ukf.initialize(
                data.F, data.H, lambda x, t: q, data.R, data.f, data.h, x=x_init, dt=dt
            )  # P init is np.eye(3)
            np.random.seed(seed + i)
            x, P = ukf.forward(y)
            # append result
            res["UKF" + str(np.diag(q).flatten())].append(RMSE(x, x_.squeeze()))

    res = {
        key: value
        for key, value in res.items()
        if (not np.isnan(value).any() and np.array(value).mean() < 1000)
    }

    # print out the results
    if verbose:
        res = dict(sorted(res.items(), key=lambda item: sum(item[1]), reverse=True))
        for k, v in res.items():
            print(
                f"RMSE {k} : {np.array(v).mean():.3f} \pm {(np.array(v).std() / np.sqrt(N)) * 1.96:.3f}"
            )


if __name__ == "__main__":
    typer.run(main)