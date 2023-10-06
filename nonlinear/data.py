import typer
import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple

plt.style.use("seaborn-v0_8-paper")


class GRW:
    def __init__(self, n: int = 1) -> None:
        """Initialize state and observation space dimensions of Gaussian Random Walk."""
        self.n = n
        self.m = n

    def f(self, x: np.array, t: float) -> np.array:
        """Transition function is identity."""
        return x

    def h(self, x: np.array, t: float) -> np.array:
        """Observation function is identity."""
        return x

    def F(self, x: np.array, t: float) -> np.array:
        """Jacobian of f is identity matrix."""
        return np.eye(self.n)

    def H(self, x: np.array, t: float) -> np.array:
        """Jacobian of h is identity matrix."""
        return np.eye(self.m)

    def Q(self, x: np.array, t: float) -> np.array:
        """Process noise is an isotropic Gaussian."""
        return self.q * np.eye(self.n)

    def R(self, x: np.array, t: float) -> np.array:
        """Measurment noise is an isotropic Gaussian."""
        return self.r * np.eye(self.m)

    def forward(
        self, T: int, dt: float = 1.0, q: float = 1.0, r: float = 1.0
    ) -> Tuple[np.array, np.array]:
        """Run dataset forward in time to produce state and measurment estimates at every time step.

        Args:
            T (int) : number of timesteps to produce data for.
            dt (float) : time change per timestep.
            q (float) : scale factor for true process noise in system.
            r (float) : scale factor for true measurement noise in system.

        Returns:
            x (np.array) : state estimates at every timestep.
            y (np.array) : measurments at every timestep.

        Raises:
            None.
        """
        self.q = q
        self.r = r
        x = np.zeros((T + 1, self.n, 1))
        y = np.zeros((T + 1, self.n, 1))
        x[0] += np.random.normal(size=(self.n, 1))  # init
        t = 0.0  # always assume starting time is 0.0
        for i in range(1, T + 1):
            x[i] = self.f(x[i - 1], t) + q * np.random.normal(size=(self.n, 1))
            y[i] = self.h(x[i], t) + r * np.random.normal(size=(self.m, 1))
            t += dt
        # return states and observations w/o initial state
        return x[1:], y[1:]


class Nonlinear:
    def __init__(self, misspecified: bool = False) -> None:
        """Initialize state and observation space dimensions of toy nonlinear system."""
        self.n = 1
        self.m = 1
        self.misspecified = misspecified

    def f(self, x: np.array, t: float) -> np.array:
        """Transition function."""
        return 0.5 * x + 25.0 * (x / (1.0 + x**2.0)) + 8.0 * np.cos(1.2 * t)

    def h(self, x: np.array, t: float) -> np.array:
        """Observation function."""
        return (x**2.0) / 20.0

    def F(self, x: np.array, t: float) -> np.array:
        """Jacobian of transition function."""
        return (0.5 * x**4.0 - 24.0 * x**2 + 25.5) / (x**2.0 + 1) ** 2.0

    def H(self, x: np.array, t: float) -> np.array:
        """Jacobian of observation function."""
        return x / 10.0

    def Q(self, x: np.array, t: float) -> np.array:
        """Process noise matrix for ideal and misspecified cases."""
        if self.misspecified:
            return np.eye(self.n) * 5.0  # * 1.0
        else:
            return np.eye(self.n) * 3.0

    def R(self, x: np.array, t: float) -> np.array:
        """Measurment noise matrix for ideal and misspecified cases."""
        return np.eye(self.m) * 2.0

    def forward(
        self, T: int, dt: float = 0.1, q: float = 3.0, r: float = 2.0
    ) -> Tuple[np.array, np.array]:
        """Run dataset forward in time to produce state and measurment estimates at every time step.

        Args:
            T (int) : number of timesteps to produce data for.
            dt (float) : time change per timestep.
            q (float) : scale factor for true process noise in system.
            r (float) : scale factor for true measurement noise in system.

        Returns:
            x (np.array) : state estimates at every timestep.
            y (np.array) : measurments at every timestep.

        Raises:
            N
        """
        x = np.zeros((T + 1, self.n, 1))
        y = np.zeros((T + 1, self.n, 1))
        x[0] += np.random.normal(size=(self.n, 1))  # initial state
        t = 0.0  # always assume starting time is 0.0
        for i in range(1, T + 1):
            x[i] = self.f(x[i - 1], t) + q * np.random.normal(size=(self.n, 1))
            y[i] = self.h(x[i], t) + r * np.random.normal(size=(self.m, 1))
            t += dt
        # return states and observations w/o initial state
        return x[1:], y[1:]


class Lorenz:
    def __init__(self, dt: float = 0.02, misspecified: bool = False) -> None:
        """Initialize state and observation space dimensions of Lorenz and dt."""
        self.n = 3
        self.m = 3
        self.dt = dt
        self.misspecified = misspecified

    def step(
        self, xyz: np.array, s: float = 10.0, r: float = 28.0, b: float = 2.667
    ) -> np.array:
        """Lorenz transition equations."""
        x, y, z = xyz
        x_dot = s * (y - x)
        y_dot = r * x - y - x * z
        z_dot = x * y - b * z
        return np.array([x_dot, y_dot, z_dot])

    def f(self, x: np.array, t: float) -> np.array:
        """Transition function with Euler or RK4 integration."""
        # if self.misspecified:
        #     return x # GRW case
        if self.misspecified:
            # Numerical integration via Euler
            if x.ndim == 3:
                for i in range(x.shape[0]):
                    x[i] = x[i] + self.step(x[i]) * self.dt
            else:
                x = x + self.step(x) * self.dt
        else:
            # Numerical integration via RK4
            if x.ndim == 3:
                for i in range(x.shape[0]):
                    k1 = self.step(x[i])
                    k2 = self.step(x[i] + 0.5 * self.dt * k1)
                    k3 = self.step(x[i] + 0.5 * self.dt * k2)
                    k4 = self.step(x[i] + self.dt * k3)
                    x[i] = x[i] + (1 / 6.0) * self.dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            else:
                k1 = self.step(x)
                k2 = self.step(x + 0.5 * self.dt * k1)
                k3 = self.step(x + 0.5 * self.dt * k2)
                k4 = self.step(x + self.dt * k3)
                x = x + (1 / 6.0) * self.dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return x

    def h(self, x: np.array, t: float) -> np.array:
        """Observation function."""
        return x

    def F(self, x: np.array, t: float) -> np.array:
        # if self.misspecified:
        #     return np.eye(self.n) # GRW case

        def jacobian(xyz: np.array, s: float = 10.0, r: float = 28.0, b: float = 2.667):
            x, y, z = xyz.flatten()
            dx = np.array([-s * self.dt, s * self.dt, 0.0])
            dy = np.array([(r - z) * self.dt, -self.dt, -x * self.dt])
            dz = np.array([y * self.dt, x * self.dt, -b * self.dt])
            return np.array([dx, dy, dz])

        return jacobian(x)

    def H(self, x: np.array, t: float) -> np.array:
        """Jacobian of observation function."""
        return np.eye(self.m)

    def Q(self, x: np.array, t: float) -> np.array:
        """Process noise matrix."""
        return np.eye(self.n) * 10.0 * self.dt

    def R(self, x: np.array, t: float) -> np.array:
        """Measurement noise matrix."""
        return np.eye(self.m) * 2.0

    def forward(
        self,
        T: int,
        q: float = 10.0,
        r: float = 2.0,
        n_em: int = 10000,
    ) -> Tuple[np.array, np.array]:
        """Run dataset forward in time to produce state and measurment estimates at every time step.

        Args:
            T (int) : number of timesteps to produce data for.
            q (float) : scale factor for true process noise in system.
            r (float) : scale factor for true measurement noise in system.
            n_em (int) : number of steps for Euler-Maruyama Integration Method

        Returns:
            x (np.array) : state estimates at every timestep.
            y (np.array) : measurments at every timestep.

        Raises:
            None.
        """
        h = self.dt / n_em
        x = np.zeros((T + 1, self.n, 1))
        y = np.zeros((T + 1, self.n, 1))
        x[0] += (
            np.random.normal(size=(self.n, 1)) + 10
        )  # init, isotropic gaussian centered on 10, 10, 10
        t = 0.0  # always assume starting time is 0.0
        for i in range(1, T + 1):
            x[i] = x[i - 1] + self.step(x[i - 1]) * h
            for j in range(1, n_em):
                x[i] = x[i] + self.step(x[i]) * h
            x[i] += q * self.dt * np.random.normal(size=(self.n, 1))
            y[i] = self.h(x[i], t) + r * np.random.normal(size=(self.m, 1))
            t += self.dt
        return x[1:], y[1:]


def main(
    seed: int = 0,
    T: int = 200,
    dt: float = 0.1,
    plot: bool = False,
):
    # linear-Gaussian data
    np.random.seed(seed)  # set random set
    grw = GRW(n=1)
    x, y = grw.forward(T)
    if plot:
        plt.figure(figsize=(6, 4))
        plt.title("Gaussian Random Walk Data")
        plt.plot(x.squeeze(), "k-", label="True State")
        plt.plot(y.squeeze(), "r.", label="Observations")
        plt.legend()

    # nonlinear data
    np.random.seed(seed)  # set random set
    nl = Nonlinear()
    x, y = nl.forward(T, dt=dt)
    if plot:
        plt.figure(figsize=(7, 5))
        plt.ylim([-30, 30])
        plt.xlim([-5, T + 5])
        plt.plot(x.squeeze(), "k-", label="True State", linewidth=2.5)
        plt.plot(y.squeeze(), "ro", label="Observations", markersize=4.2)
        legend = plt.legend(
            loc="upper center",
            fontsize=13,
            ncols=2,
            frameon=False,
            bbox_to_anchor=(0.5, 1.15),
        )
        for handle in legend.legendHandles:
            handle.set_linewidth(3.5)  # Optional: Adjust the handle line width
        plt.xlabel("Timestep", fontsize=13, fontweight="bold")
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)

        dpi = 600
        plt.savefig("data.png", dpi=dpi)

    if plot:
        # stochastic lorenz attractor system
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), subplot_kw={"projection": "3d"})

        for i, ax in enumerate(axes):
            np.random.seed(seed + i)  # Set a different seed for each instance
            lorenz = Lorenz()
            x, y = lorenz.forward(T)

            ax.plot(*x.squeeze().T, "k-", lw=1.0, label="True State")
            ax.plot(*y.squeeze().T, "r.", label="Observations")
            ax.set_xlabel("X1")
            ax.set_ylabel("X2")
            ax.set_zlabel("X3")
            ax.legend()

    if plot:
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    typer.run(main)
