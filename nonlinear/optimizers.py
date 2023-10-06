import abc
from typing import Callable

import numpy as np
from numpy.linalg import inv
from helper import symmetric_diag


class Optimizer(abc.ABC):
    def __init__(self):
        super(Optimizer, self).__init__()

    @abc.abstractmethod
    def step(self):
        """Take a single step."""

    def initialize(
        self,
        theta: np.array,
        h: Callable,
        H: Callable,
    ):
        self.t = 0
        self.theta = theta
        self.h = h
        self.H = H

    def forward(self, y: np.array, return_steps: bool = False):
        """Compute gradient of Isotropic Gaussian likelihood.
        Propagate the optimizer forward until convergence or stopping criteria."""
        steps = [self.theta]
        for _ in range(self.k):
            v = y - self.h(self.theta)
            g = -self.H(self.theta).T @ v
            self.step(g)
            if return_steps:
                steps.append(self.theta)
        if return_steps:
            return self.theta, steps
        else:
            return self.theta


class Adam(Optimizer):
    """Standard implemention of Adam in numpy using Gaussian likelihood."""

    def __init__(
        self,
        k: int,
        alpha: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.k = k
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

    def initialize(
        self,
        theta: np.array,
        h: Callable,
        H: Callable,
    ):
        self.m = 0.0
        self.v = 0.0
        self.t = 0
        self.theta = theta
        self.h = h
        self.H = H

    def step(self, g: np.array):
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * g
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * g**2.0
        alpha = self.alpha * np.sqrt(1.0 - self.beta2) / (1 - self.beta1)
        self.theta = self.theta - alpha * self.m / (np.sqrt(self.v) + self.eps)
        self.t += 1


class AMSgrad(Optimizer):
    """Standard implemention of AMSgrad in numpy using Gaussian likelihood."""

    def __init__(
        self,
        k: int,
        alpha: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.k = k
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

    def initialize(
        self,
        theta: np.array,
        h: Callable,
        H: Callable,
    ):
        self.m = 0.0
        self.v = 0.0
        self.vhat = -np.inf
        self.t = 0
        self.theta = theta
        self.h = h
        self.H = H

    def step(self, g: np.array):
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * g
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * g**2.0
        self.vhat = max(self.vhat, self.v)
        self.theta = self.theta - self.alpha * self.m / (np.sqrt(self.vhat) + self.eps)
        self.t += 1


class Adagrad(Optimizer):
    """Standard implemention of Adagrad in numpy using Gaussian likelihood."""

    def __init__(
        self,
        k: int,
        alpha: float = 1e-3,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.k = k
        self.alpha = alpha
        self.eps = eps

    def initialize(
        self,
        theta: np.array,
        h: Callable,
        H: Callable,
    ):
        self.G = np.zeros_like(theta)
        self.t = 0
        self.theta = theta
        self.h = h
        self.H = H

    def step(self, g: np.array):
        self.G += g**2.0
        self.theta = self.theta - (self.alpha * g) / np.sqrt(self.G + self.eps)
        self.t += 1


class RMSprop(Optimizer):
    """Standard implemention of RMSprop in numpy using Gaussian likelihood."""

    def __init__(
        self,
        k: int,
        alpha: float = 1e-3,
        gamma: float = 0.9,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.k = k
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def step(self, g: np.array):
        if self.t == 0:
            self.G = g**2.0
        self.G = self.gamma * self.G + (1 - self.gamma) * g**2.0
        self.theta = self.theta - (self.alpha * g) / np.sqrt(self.G + self.eps)
        self.t += 1


class Adadelta(Optimizer):
    """Standard implemention of Adadelta in numpy using Gaussian likelihood."""

    def __init__(
        self,
        k: int,
        gamma: float = 0.9,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.k = k
        self.gamma = gamma
        self.eps = eps

    def step(self, g: np.array):
        if self.t == 0:
            self.G = g**2.0
            self.delta = np.ones_like(self.theta)
        self.G = self.gamma * self.G + (1 - self.gamma) * g**2.0
        delta = -(self.delta * g) / self.G
        self.theta = self.theta + delta
        self.delta = self.gamma * self.delta + (1 - self.gamma) * delta**2.0
        self.t += 1


class GD(Optimizer):
    """Standard implemention of gradient descent in numpy using Gaussian likelihood."""

    def __init__(
        self,
        k: int,
        alpha: float = 1e-3,
    ):
        super().__init__()
        self.k = k
        self.alpha = alpha

    def step(self, g: np.array):
        self.theta = self.theta - self.alpha * g
        self.t += 1


class Santos(Optimizer):
    """Optimizer inspired by Santos (1996) where we convert from a prior to a learning rate matrix."""

    def __init__(self, P: np.array, k: int, recompute: bool = False):
        super().__init__()
        self.P = P
        self.k = k
        self.recompute = (
            recompute  # recompute the learning rate matrix at every timestep
        )

    def initialize(
        self,
        theta: np.array,
        h: Callable,
        H: Callable,
    ):
        self.t = 0
        self.theta = theta
        self.h = h
        self.H = H
        self.M = self.computeM()

    def computeM(self):
        """Compute the learning rate matrix via Santos (1996)."""
        X, p = symmetric_diag(
            inv(self.P),
            self.H(self.theta).T
            @ self.H(self.theta),  # arbitrarily set the measurement noise to identity
        )
        lam = np.where(
            np.equal(p, 0.0),
            np.ones_like(p),
            np.reciprocal(p)
            * (1 - (1 + p) ** (-1.0 / self.k)),  # TODO : why is q = 1 in Santos
        )
        M = X @ np.diag(lam) @ X.T
        return M

    def step(self, g: np.array):
        self.theta = self.theta - self.M @ g
        if self.recompute:
            self.M = self.computeM()
