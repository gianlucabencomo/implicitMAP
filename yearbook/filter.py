import jax
import jax.numpy as jnp
import jax.random as jr

import optax

from typing import Callable, Tuple, Any
from functools import partial

Optimizer = Any


class MAP:
    def __init__(
        self,
        steps: int,
        optimizer: Optimizer,
    ) -> None:
        """Initialize steps and optimizer to be used during update step."""
        super().__init__()
        self.steps = steps
        self.optimizer = optimizer

    def initialize(
        self,
        f: Callable,
        loss: Callable,
        w: jax.Array,
    ) -> jax.Array:
        """Initialize forward pass function f, loss function, initial weights, and timestep."""
        self.f = f
        self.loss = partial(loss, f=f)
        self.w = w
        self.t = 0
        return self.w

    def predict(self, F: Callable = lambda x, t: x) -> jax.Array:
        """Apply dynamics to weights."""
        self.w = F(self.w, self.t)
        return self.w

    def update(self, X: jax.Array, y: jax.Array) -> jax.Array:
        """Perform optimization over data (X, y)."""
        opt_state = self.optimizer.init(self.w)

        @jax.jit
        def step(w, opt_state):
            g = jax.grad(self.loss)(w, X, y)
            updates, opt_state = self.optimizer.update(g, opt_state, w)
            w = optax.apply_updates(w, updates)
            return w, opt_state

        for _ in range(self.steps):
            self.w, opt_state = step(self.w, opt_state)
        self.t += 1
        return self.w

    def forward(self, X: list, y: list) -> jax.Array:
        """Run filter forward for every time step in (X, y)."""
        w = []
        for i in range(len(y)):
            self.predict()
            self.update(X[i], y[i])
            w.append(self.w)
        return w


class Bootstrap:
    def __init__(self, n_particles: int = 1000, eps: float = 1e-8) -> None:
        super().__init__()
        """Initialize number of particles and epsilon term for numerical stability."""
        self.n_particles = n_particles
        self.eps = eps

    def initialize(
        self,
        key: jr.PRNGKey,
        Q: Callable,
        f: Callable,
        loss: Callable,
        w: jax.Array,
    ) -> jax.Array:
        """Initialize particle filter parameters."""
        self.key = key
        self.f = f
        self.Q = Q  # process noise
        self.loss = partial(loss, f=f)
        self.w = jnp.tile(jnp.expand_dims(w, axis=0), (self.n_particles, 1))
        self.t = 0
        return self.w

    def predict(self) -> jax.Array:
        """Apply transition distribution to move particles forward."""
        self.w = self.w + jr.normal(self.key, shape=self.w.shape) * self.Q(
            self.w, self.t
        )
        self.key, _ = jr.split(self.key)
        return self.w

    def update(self, X: jax.Array, y: jax.Array) -> jax.Array:
        """Weight particles via likelihood and resample."""
        weights = jnp.zeros((self.n_particles,))
        for i in range(self.n_particles):
            weights = weights.at[i].set(1.0 / self.loss(self.w[i], X, y))
        weights = weights + self.eps  # stabilize the weights
        weights = weights / weights.sum()
        self.weights = weights
        w_max = self.w[jnp.argmax(self.weights)]
        inds = jr.choice(
            self.key,
            self.n_particles,
            shape=(self.n_particles,),
            replace=False,
            p=weights,
        )
        self.w = self.w[inds]
        self.key, _ = jr.split(self.key)
        self.t += 1
        return w_max

    def forward(self, X: list, y: list) -> jax.Array:
        """Run filter forward for every time step in (X, y)."""
        w = []
        for i in range(len(y)):
            self.predict()
            w_max = self.update(X[i], y[i])
            w.append(w_max)
        return w


class VKF:
    def __init__(
        self,
        steps: int,
        optimizer,
    ) -> None:
        """Initialize steps and optimizer to be used during update step."""
        super().__init__()
        self.steps = steps
        self.optimizer = optimizer

    def initialize(
        self,
        key: jr.PRNGKey,
        q: float,
        f: Callable,
        w: jax.Array,
    ) -> jax.Array:
        """Initialize VKF parameters."""
        self.key = key
        self.f = f
        self.q = q
        self.w = w
        self.t = 0
        return self.w

    def loss(self, w, X, y: jax.Array) -> jax.Array:
        """VKF loss function."""
        n = X.shape[0]
        z = self.f(w, X)
        return (
            jnp.mean(optax.sigmoid_binary_cross_entropy(z, y))
            + (1.0 / (2 * n * self.q)) * ((self.w_pred - w) ** 2).sum()
        )

    def predict(self) -> jax.Array:
        """Apply transition distribution."""
        self.w_pred = self.w + jr.normal(self.key, shape=self.w.shape) * self.q
        self.key, _ = jr.split(self.key)
        return self.w

    def update(self, X: jax.Array, y: jax.Array) -> jax.Array:
        """Update point estimate via optimizing VKF loss to convergence."""
        opt_state = self.optimizer.init(self.w_pred)
        self.w = self.w_pred

        @jax.jit
        def step(w, opt_state):
            g = jax.grad(self.loss)(w, X, y)
            updates, opt_state = self.optimizer.update(g, opt_state, w)
            w = optax.apply_updates(w, updates)
            return w, opt_state

        for _ in range(self.steps):
            self.w, opt_state = step(self.w, opt_state)
        self.t += 1
        return self.w

    def forward(self, X: list, y: list) -> jax.Array:
        """Run filter forward for every time step in (X, y)."""
        w = []
        for i in range(len(y)):
            self.predict()
            self.update(X[i], y[i])
            w.append(self.w)
        return w
