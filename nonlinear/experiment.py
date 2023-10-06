import typer
import numpy as np

from data import Lorenz, Nonlinear
from filter import KalmanFilter, MAP, IteratedEKF, UnscentedKF, Bootstrap
from helper import RMSE
from optimizers import Adam, Santos, Adagrad, RMSprop, Adadelta, AMSgrad, GD
from config import get_config


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


def init_dict(
    optim: list,
    steps: list,
    adagrad_hyp: list = None,
    adam_hyp: list = None,
    rmsprop_hyp: list = None,
    ams_hyp: list = None,
    gd_hyp: list = None,
):
    """Set up dictionary for running experiments."""
    res = {"EKF": []}
    res["UKF"] = []
    # res["PF"] = []
    for i in steps:
        if i == 1:
            continue  # this is just an EKF
        res["IEKF|" + str(i)] = []
    if "Santos" in optim:
        for i in steps:
            for j in [True, False]:
                res["Santos MAP|" + str(i) + "|" + str(j)] = []
    if "Adadelta" in optim:
        for i in steps:
            res["Adadelta MAP|" + str(i)] = []
    if "Adagrad" in optim:
        for i in steps:
            for alpha in adagrad_hyp:
                res["Adagrad MAP|" + str(i) + "|" + str(alpha)] = []
    if "GD" in optim:
        for i in steps:
            for alpha in gd_hyp:
                res["GD MAP|" + str(i) + "|" + str(alpha)] = []
    if "RMSprop" in optim:
        for i in steps:
            for alpha, gamma in rmsprop_hyp:
                res["RMSprop MAP|" + str(i) + "|" + str(alpha) + "|" + str(gamma)] = []
    if "Adam" in optim and adam_hyp != None:
        for i in steps:
            for alpha, beta1, beta2 in adam_hyp:
                res[
                    "Adam MAP|"
                    + str(i)
                    + "|"
                    + str(alpha)
                    + "|"
                    + str(beta1)
                    + "|"
                    + str(beta2)
                ] = []
    if "AMSgrad" in optim and ams_hyp != None:
        for i in steps:
            for alpha, beta1, beta2 in adam_hyp:
                res[
                    "AMSgrad MAP|"
                    + str(i)
                    + "|"
                    + str(alpha)
                    + "|"
                    + str(beta1)
                    + "|"
                    + str(beta2)
                ] = []
    return res


def main(
    N: int = 100,
    seed: int = 0,
    T: int = 200,
    verbose: bool = False,
    system: str = "lorenz",
    misspecified: bool = False,
):
    # select optimizers and get configs
    opt_list = ["Adam", "RMSprop", "Adagrad", "Adadelta", "GD"]
    steps, adagrad_hyp, adam_hyp, rmsprop_hyp, ams_hyp, gd_hyp = get_config(system)

    # init dict
    res = init_dict(
        opt_list, steps, adagrad_hyp, adam_hyp, rmsprop_hyp, ams_hyp, gd_hyp
    )

    # init data + other configs
    data, n, m = init_data(system, misspecified)
    dt = 0.02 if system == "lorenz" else 0.1
    n_particles = 1000

    if verbose:
        print(
            f"Testing {len(res)} models and baselines for {N} random experiments (Q = {data.Q(0,0)})..."
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

        # ! Run Kalman Filter baseline
        kf = KalmanFilter(n, m)
        kf.initialize(
            data.F, data.H, data.Q, data.R, data.f, data.h, x=x_init, dt=dt
        )  # P init is np.eye(3)
        np.random.seed(seed + i)
        x, P = kf.forward(y)
        # append result
        res["EKF"].append(RMSE(x, x_.squeeze()))

        # ! Run Unscented Kalman Filter baseline
        ukf = UnscentedKF(n, m)
        ukf.initialize(
            data.F, data.H, data.Q, data.R, data.f, data.h, x=x_init, dt=dt
        )  # P init is np.eye(3)
        np.random.seed(seed + i)
        x, P = ukf.forward(y)
        # append result
        res["UKF"].append(RMSE(x, x_.squeeze()))

        # ! Run Particle  Filter baseline
        if system == "lorenz":
            x_pf_init = np.random.normal(
                10, 1, size=(n_particles, n, 1)
            )  # sample from isotropic gaussian prior centered at 10
        else:
            x_pf_init = None
        pf = Bootstrap(n, m, n_particles=n_particles)
        pf.initialize(data.Q, data.R, data.f, data.h, x=x_pf_init, dt=dt)
        np.random.seed(seed + i)
        x, _ = pf.forward(y)
        # get mean
        x = x.mean(axis=1)
        # append result
        res["PF"].append(RMSE(x, x_.squeeze()))

        # ! Run Iterated Extended Kalman Filter baseline
        for i in steps:
            if i == 1:
                continue  # this is just an EKF
            iekf = IteratedEKF(n, m, n_steps=i)
            iekf.initialize(
                data.F,
                data.H,
                data.Q,
                data.R,
                data.f,
                data.h,
                x=x_init,
                dt=dt,  # P init is np.eye(3)
            )
            np.random.seed(seed + i)
            x, P = iekf.forward(y)
            # append result
            res["IEKF|" + str(i)].append(RMSE(x, x_.squeeze()))

        # ! run MAP filter for Santos
        if "Santos" in opt_list:
            P = np.eye(n)
            for i in steps:
                for j in [True, False]:
                    optimizer = Santos(P, i, j)
                    map = MAP(n, m, optimizer)
                    map.initialize(data.H, data.f, data.h, x=x_init, dt=dt)
                    np.random.seed(seed + i)
                    x = map.forward(y)
                    res["Santos MAP|" + str(i) + "|" + str(j)].append(
                        RMSE(x, x_.squeeze())
                    )

        # ! run MAP filter for Adam
        if "Adam" in opt_list:
            for i in steps:
                for alpha, beta1, beta2 in adam_hyp:
                    # Running Adam
                    optimizer = Adam(k=i, alpha=alpha, beta1=beta1, beta2=beta2)
                    map = MAP(n, m, optimizer)
                    map.initialize(data.H, data.f, data.h, x=x_init, dt=dt)
                    np.random.seed(seed + i)
                    x = map.forward(y)
                    res[
                        "Adam MAP|"
                        + str(i)
                        + "|"
                        + str(alpha)
                        + "|"
                        + str(beta1)
                        + "|"
                        + str(beta2)
                    ].append(RMSE(x, x_.squeeze()))

        # ! run MAP filter for AMSgrad
        if "AMSgrad" in opt_list:
            for i in steps:
                for alpha, beta1, beta2 in adam_hyp:
                    # Running Adam
                    optimizer = AMSgrad(k=i, alpha=alpha, beta1=beta1, beta2=beta2)
                    map = MAP(n, m, optimizer)
                    map.initialize(data.H, data.f, data.h, x=x_init, dt=dt)
                    np.random.seed(seed + i)
                    x = map.forward(y)
                    res[
                        "AMSgrad MAP|"
                        + str(i)
                        + "|"
                        + str(alpha)
                        + "|"
                        + str(beta1)
                        + "|"
                        + str(beta2)
                    ].append(RMSE(x, x_.squeeze()))

        # ! run MAP filter for Adagrad
        if "Adagrad" in opt_list:
            for i in steps:
                for alpha in adagrad_hyp:
                    optimizer = Adagrad(k=i, alpha=alpha)
                    map = MAP(n, m, optimizer)
                    map.initialize(data.H, data.f, data.h, x=x_init, dt=dt)
                    np.random.seed(seed + i)
                    x = map.forward(y)
                    res["Adagrad MAP|" + str(i) + "|" + str(alpha)].append(
                        RMSE(x, x_.squeeze())
                    )

        # ! run MAP filter for Adadelta
        if "Adadelta" in opt_list:
            for i in steps:
                optimizer = Adadelta(k=i)
                map = MAP(n, m, optimizer)
                map.initialize(data.H, data.f, data.h, x=x_init, dt=dt)
                np.random.seed(seed + i)
                x = map.forward(y)
                res["Adadelta MAP|" + str(i)].append(RMSE(x, x_.squeeze()))

        # ! run MAP filter for GD
        if "GD" in opt_list:
            for i in steps:
                for alpha in gd_hyp:
                    optimizer = GD(k=i, alpha=alpha)
                    map = MAP(n, m, optimizer)
                    map.initialize(data.H, data.f, data.h, x=x_init, dt=dt)
                    np.random.seed(seed + i)
                    x = map.forward(y)
                    res["GD MAP|" + str(i) + "|" + str(alpha)].append(
                        RMSE(x, x_.squeeze())
                    )

        # ! run MAP filter for RMSprop
        if "RMSprop" in opt_list:
            for i in steps:
                for alpha, gamma in rmsprop_hyp:
                    optimizer = RMSprop(k=i, alpha=alpha, gamma=gamma)
                    map = MAP(n, m, optimizer)
                    map.initialize(data.H, data.f, data.h, x=x_init, dt=dt)
                    np.random.seed(seed + i)
                    x = map.forward(y)
                    res[
                        "RMSprop MAP|" + str(i) + "|" + str(alpha) + "|" + str(gamma)
                    ].append(RMSE(x, x_.squeeze()))

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
