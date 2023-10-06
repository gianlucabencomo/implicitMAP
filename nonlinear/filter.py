import abc
import numpy as np
from typing import Callable, Any
from functools import partial

from numpy.linalg import inv, cholesky
from scipy.stats import norm, multivariate_normal

from typing import Tuple

Optimizer = Any


class Filter(abc.ABC):
    def __init__(self) -> None:
        super(Filter, self).__init__()

    @abc.abstractmethod
    def initialize(
        self,
        F: Callable,
        H: Callable,
        Q: np.array,
        R: np.array,
        f: Callable,
        h: Callable,
        x: np.array,
        P: np.array,
    ) -> Tuple[np.array, np.array]:
        """Initialize and return filter parameters."""

    @abc.abstractmethod
    def predict(self) -> Tuple[np.array, np.array]:
        """Make a prediction for one timestep."""

    @abc.abstractmethod
    def update(self, y: np.array) -> Tuple[np.array, np.array]:
        """Compute new parameters given previous parameters and current datasets."""

    @abc.abstractmethod
    def forward(self, y: np.array) -> Tuple[np.array, np.array]:
        """Propagate filter forward in time given initial starting conditions."""


class KalmanFilter(Filter):
    def __init__(self, n: int, m: int) -> None:
        super().__init__()
        """Set state and observation space dimensions."""
        self.n = n
        self.m = m

    def initialize(
        self,
        F: Callable,
        H: Callable,
        Q: Callable,
        R: Callable,
        f: Callable = None,
        h: Callable = None,
        x: np.array = None,
        P: np.array = None,
        dt: float = 1.0,
    ) -> Tuple[np.array, np.array]:
        """Initialize initial filter configuration.

        Args:
            F (Callable) : Jacobian of transition function.
            H (Callable) : Jacobian of measurement function.
            Q (Callable) : process noise function.
            R (Callable) : measurement noise function.
            f (Callable) : transition function.
            h (Callable) : measurment function.
            x (np.array) : initial state.
            P (np.array) : initial state uncertainty.
            dt (float) : change in time per timestep.

        Returns:
            x (np.array) : state
            P (np.array) : state uncertainty.

        Raises:
            None.
        """
        self.f = f
        self.F = F
        self.h = h
        self.H = H
        self.Q = Q
        self.R = R
        self.P = np.eye(self.n) if P is None else P
        self.x = np.zeros((self.n, 1)) if x is None else x
        self.t = 0.0
        self.dt = dt
        return self.x, self.P

    def predict(self) -> Tuple[np.array, np.array]:
        """Apply Kalman filtering predict step for state and uncertainty."""
        self.x = self.f(self.x, self.t)
        self.P = self.F(self.x, self.t) @ self.P @ self.F(self.x, self.t).T + self.Q(
            self.x, self.t
        )
        return self.x, self.P

    def update(self, y: np.array) -> Tuple[np.array, np.array]:
        """Apply Kalman filtering update step for state and uncertainty."""
        v = y - self.h(self.x, self.t)
        S = self.H(self.x, self.t) @ self.P @ self.H(self.x, self.t).T + self.R(
            self.x, self.t
        )
        K = self.P @ self.H(self.x, self.t).T @ inv(S)
        self.x = self.x + K @ v
        self.P = self.P - K @ S @ K.T
        self.t += self.dt
        return self.x, self.P

    def forward(self, y: np.array) -> Tuple[np.array, np.array]:
        """Run Kalman filter forward in time."""
        x, P = [], []
        for i in y:
            self.predict()
            self.update(i)
            x.append(self.x)
            P.append(self.P)
        return np.array(x).squeeze(), np.array(P)

    def riccati(self, tol: float = 1e-8, maxiter: int = 1000) -> np.array:
        """Find the Kalman filter steady state via Riccati equations."""
        diff, i = np.inf, 0
        while i < maxiter and diff > tol:
            S = self.H(self.x, self.t) @ self.P @ self.H(self.x, self.t).T + self.R(
                self.x, self.t
            )
            K = self.P @ self.H(self.x, self.t).T @ inv(S)
            P = (
                self.F(self.x, self.t) @ self.P @ self.F(self.x, self.t).T
                + self.Q(self.x, self.t)
                - K @ S @ K.T
            )
            diff = np.abs(P - self.P).sum()
            i += 1
            self.P = P
        Phat = self.P  # stable value of covariance from predict step
        self.P = self.P - self.Q(self.x, self.t)
        return Phat


class IteratedEKF(Filter):
    def __init__(self, n: int, m: int, n_steps: int = 1) -> None:
        super().__init__()
        """Set state and observation space dimensions and the number of iteration steps."""
        self.n = n
        self.m = m
        self.n_steps = n_steps

    def initialize(
        self,
        F: Callable,
        H: Callable,
        Q: Callable,
        R: Callable,
        f: Callable = None,
        h: Callable = None,
        x: np.array = None,
        P: np.array = None,
        dt: float = 1.0,
    ) -> Tuple[np.array, np.array]:
        """Initialize initial filter configuration.

        Args:
            F (Callable) : Jacobian of transition function.
            H (Callable) : Jacobian of measurement function.
            Q (Callable) : process noise function.
            R (Callable) : measurement noise function.
            f (Callable) : transition function.
            h (Callable) : measurment function.
            x (np.array) : initial state.
            P (np.array) : initial state uncertainty.
            dt (float) : change in time per timestep.

        Returns:
            x (np.array) : state
            P (np.array) : state uncertainty.

        Raises:
            None.
        """
        self.f = f
        self.F = F
        self.h = h
        self.H = H
        self.Q = Q
        self.R = R
        self.P = np.eye(self.n) if P is None else P
        self.x = np.zeros((self.n, 1)) if x is None else x
        self.t = 0.0
        self.dt = dt
        return self.x, self.P

    def predict(self) -> Tuple[np.array, np.array]:
        """Apply Kalman filtering predict step for state and uncertainty."""
        self.x = self.f(self.x, self.t)
        self.P = self.F(self.x, self.t) @ self.P @ self.F(self.x, self.t).T + self.Q(
            self.x, self.t
        )
        return self.x, self.P

    def update(self, y: np.array) -> Tuple[np.array, np.array]:
        """Apply Kalman filtering update step for state and uncertainty,
        with iterative refinement of linearization."""
        m_ = self.x
        for _ in range(self.n_steps):
            v = y - self.h(self.x, self.t) - self.H(self.x, self.t) @ (m_ - self.x)
            S = self.H(self.x, self.t) @ self.P @ self.H(self.x, self.t).T + self.R(
                self.x, self.t
            )
            K = self.P @ self.H(self.x, self.t).T @ inv(S)
            self.x = m_ + K @ v
        self.P = self.P - K @ S @ K.T
        self.t += self.dt
        return self.x, self.P

    def forward(self, y: np.array) -> Tuple[np.array, np.array]:
        """Run Kalman filter forward in time."""
        x, P = [], []
        for i in y:
            self.predict()
            self.update(i)
            x.append(self.x)
            P.append(self.P)
        return np.array(x).squeeze(), np.array(P)


class UnscentedKF(Filter):
    def __init__(self, n: int, m: int, alpha: int = 1.0) -> None:
        """Set state and observation space dimensions and some default values for UKF.
        See Särkkä (2023) for more details."""
        super().__init__()
        self.n = n
        self.m = m
        # UKF params
        self.alpha = alpha
        self.kappa = 3.0 - n
        self.lam = (self.alpha**2) * (self.n + self.kappa) - self.n
        # UKF init
        wm = np.ones((2 * self.n + 1,) + (self.n, 1)) * (1 / (2 * (self.n + self.lam)))
        wc = np.ones((2 * self.n + 1,) + (self.n, 1)) * (1 / (2 * (self.n + self.lam)))
        wm[0] = self.lam / (self.n + self.lam)
        wc[0] = wm[0] + (1 - self.alpha**2)
        self.wm = wm / wm.sum()
        self.wc = wc / wc.sum()

    def initialize(
        self,
        F: Callable,
        H: Callable,
        Q: Callable,
        R: Callable,
        f: Callable = None,
        h: Callable = None,
        x: np.array = None,
        P: np.array = None,
        dt: float = 1.0,
    ) -> Tuple[np.array, np.array]:
        """Initialize initial filter configuration.

        Args:
            F (Callable) : Jacobian of transition function.
            H (Callable) : Jacobian of measurement function.
            Q (Callable) : process noise function.
            R (Callable) : measurement noise function.
            f (Callable) : transition function.
            h (Callable) : measurment function.
            x (np.array) : initial state.
            P (np.array) : initial state uncertainty.
            dt (float) : change in time per timestep.

        Returns:
            x (np.array) : state
            P (np.array) : state uncertainty.

        Raises:
            None.
        """
        self.f = f
        self.F = F
        self.h = h
        self.H = H
        self.Q = Q
        self.R = R
        self.P = np.eye(self.n) if P is None else P
        self.x = np.zeros((self.n, 1)) if x is None else x
        self.t = 0.0
        self.dt = dt
        return self.x, self.P

    def sigma(self) -> np.array:
        """Compute sigma points for unscented transform."""
        P_sqrt = cholesky(self.P).reshape(self.n, self.n, 1)
        sigma = np.zeros((2 * self.n + 1,) + self.x.shape)
        sigma[0] = self.x
        for i in range(1, self.n + 1):
            sigma[i] = self.x + np.sqrt(self.n + self.lam) * P_sqrt[:, i - 1]
            sigma[self.n + i] = self.x - np.sqrt(self.n + self.lam) * P_sqrt[:, i - 1]
        return sigma

    def predict(self) -> Tuple[np.array, np.array]:
        """Apply unscented Kalman filter predict step."""
        sigma = self.sigma()
        sigma = self.f(sigma, self.t)
        self.x = (self.wm * sigma).sum(axis=0)
        self.P = 0.0
        for i in range(2 * self.n + 1):
            self.P += self.wc[i] * (sigma[i] - self.x) @ (sigma[i] - self.x).T
        self.P += self.Q(self.x, self.t)
        return self.x, self.P

    def update(self, y: np.array) -> Tuple[np.array, np.array]:
        """Apply unscented Kalman filter update step."""
        sigma = self.sigma()
        yhat = self.h(sigma, self.t)
        mu = (self.wm * yhat).sum(axis=0)
        S = 0.0
        for i in range(2 * self.n + 1):
            S += self.wc[i] * (yhat[i] - mu) @ (yhat[i] - mu).T
        S += self.R(self.x, self.t)
        C = 0.0
        for i in range(2 * self.n + 1):
            C += self.wc[i] * (sigma[i] - self.x) @ (yhat[i] - mu).T
        K = C @ inv(S)
        self.x = self.x + K @ (y - mu)
        self.P = self.P - K @ S @ K.T
        self.t += self.dt
        return self.x, self.P

    def forward(self, y: np.array) -> Tuple[np.array, np.array]:
        """Run unscented Kalman filter forward in time."""
        x, P = [], []
        for i in y:
            self.predict()
            self.update(i)
            x.append(self.x)
            P.append(self.P)
        return np.array(x).squeeze(), np.array(P)


class Bootstrap(Filter):
    def __init__(
        self, n: int, m: int, n_particles: int = 1000, eps: float = 1e-8
    ) -> None:
        """Set state and observation space dimensions, number of particles, and numerical stability factor."""
        super().__init__()
        self.n = n
        self.m = m
        self.n_particles = n_particles
        self.eps = eps

    def initialize(
        self,
        Q: Callable,
        R: Callable,
        f: Callable = None,
        h: Callable = None,
        x: np.array = None,
        dt: float = 1.0,
    ) -> Tuple[np.array, None]:
        """Initialize initial filter configuration.

        Args:
            f (Callable) : transition function.
            h (Callable) : measurment function.
            Q (Callable) : process noise function.
            R (Callable) : measurement noise function.
            x (np.array) : initial state.
            P (np.array) : initial state uncertainty.
            dt (float) : change in time per timestep.

        Returns:
            x (np.array) : state
            P (empty) : return None.

        Raises:
            None.
        """
        self.f = f
        self.h = h
        self.Q = Q
        self.R = R
        self.x = (
            np.random.normal(size=(self.n_particles, self.n, 1)) if x is None else x
        )
        self.t = 0.0
        self.dt = dt
        return self.x, None

    def predict(self) -> Tuple[np.array, None]:
        """Apply transition distribution to particles for the predict step."""
        if self.n > 1:
            self.x = self.f(self.x, self.t) + np.random.multivariate_normal(
                mean=np.zeros((self.n)),
                cov=self.Q(self.x, self.t),
                size=(self.n_particles,),
            ).reshape(self.x.shape)
        else:
            self.x = self.f(self.x, self.t) + np.random.normal(
                size=(self.n_particles, self.n, 1),
            ) * self.Q(self.x, self.t)
        return self.x, None

    def update(self, y: np.array) -> Tuple[np.array, None]:
        """Calculate weights via likelihood and resample."""
        # calculate weights under gaussian likelihood
        if self.n == 1:
            weights = norm.pdf(
                y, loc=self.h(self.x, self.t), scale=self.R(self.x, self.t)
            ).squeeze()
        else:
            weights = np.zeros((self.n_particles,))
            for i in range(self.n_particles):
                mv = multivariate_normal(
                    self.h(self.x[i], self.t).squeeze(), self.R(self.x[i], self.t)
                )
                weights[i] = mv.pdf(y.squeeze())
        weights = weights + self.eps  # stabilize the weights
        weights = weights / weights.sum()
        inds = np.random.choice(
            self.n_particles, size=self.n_particles, replace=True, p=weights
        )
        self.x = self.x[inds]
        self.t += self.dt
        self.weights = weights
        return self.x, None

    def effective(self):
        """Effective sample size."""
        if hasattr(self, "weights"):
            return 1.0 / (self.weights**2.0).sum()
        else:
            return 0.0

    def forward(self, y: np.ndarray) -> Tuple[np.array, None]:
        """Run particle filter forward in time."""
        x = []
        for i in y:
            self.predict()
            self.update(i)
            x.append(self.x)
        return np.array(x).squeeze(), None


class MAP(Filter):
    def __init__(
        self,
        n: int,
        m: int,
        optimizer: Optimizer,
    ) -> None:
        """Set state and observation space dimensions and the optimizer to be used."""
        super().__init__()
        self.n = n
        self.m = m
        self.optimizer = optimizer

    def initialize(
        self,
        H: Callable,
        f: Callable,
        h: Callable,
        x: np.array = None,
        dt: float = 1.0,
    ) -> np.array:
        """Initialize initial filter configuration.

        Args:
            H (Callable) : Jacobian of measurement function.
            f (Callable) : transition function.
            h (Callable) : measurment function.
            x (np.array) : initial state.
            dt (float) : change in time per timestep.

        Returns:
            x (np.array) : state

        Raises:
            None.
        """
        self.f = f
        self.h = h
        self.H = H
        self.x = np.zeros((self.n, 1)) if x is None else x
        self.t = 0.0
        self.dt = dt
        return self.x

    def predict(self) -> np.array:
        """Apply transition function for predict step."""
        self.x = self.f(self.x, self.t)
        return self.x

    def update(self, y: np.array, return_steps: bool = False) -> np.array:
        """Run optimization with k steps as a surrogate for the update step of a standard filter."""
        h, H = partial(self.h, t=self.dt), partial(self.H, t=self.dt)
        self.optimizer.initialize(self.x, h, H)
        if return_steps:
            self.x, steps = self.optimizer.forward(y, return_steps)
        else:
            self.x = self.optimizer.forward(y)
        self.t += self.dt
        if return_steps:
            return self.x, steps
        else:
            return self.x

    def forward(self, y: np.array, return_steps: bool = False) -> np.array:
        """Run MAP filter forward in time."""
        x = []
        if return_steps:
            steps = []
        for i in range(y.shape[0]):
            self.predict()
            if return_steps:
                _, s = self.update(y[i], return_steps)
            else:
                self.update(y[i], return_steps)
            x.append(self.x)
            if return_steps:
                steps.append(s)
        if return_steps:
            return np.array(x).squeeze(), np.array(steps).squeeze()
        else:
            return np.array(x).squeeze()
