import typer
from typing import Callable, Tuple, Any

import jax
import jax.random as jr
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax._src.config import config

import optax
from optax import sgd, adam

from model import CNN
from data import Yearbook
from filter import MAP, Bootstrap, VKF

config.update("jax_enable_x64", True)

Optimizer = Any


def init_nn(key: jr.KeyArray, in_dim: int) -> Tuple[Callable, jax.Array]:
    """Initialize CNN used in yearbook experiments.

    Args:
        key (jr.KeyArray) : jax.random key.
        in_dim : input dimensions for CNN.

    Returns:
        f : jitted forward-pass function of flax CNN.
        w (jax.Array) : flattened weights of CNN.

    Raises:
        None.
    """
    model = CNN()
    params = model.init(key, jnp.zeros(in_dim))["params"]
    w, unflatten = ravel_pytree(params)
    w = w.astype(jnp.float64)

    @jax.jit
    def f(w, x):
        return jnp.squeeze(model.apply({"params": unflatten(w)}, x), axis=-1)

    return f, w


def init_w(
    key: jr.KeyArray,
    w: jax.Array,
    X: jax.Array,
    y: jax.Array,
    f: Callable,
    L: Callable,
    optimizer: Optimizer,
    steps: int = 200,
    batch_size: int = 64,
) -> jax.Array:
    """Pre-train the initial network via SGD and return weight initialization.

    Args:
        key (jr.KeyArray) : jax.random key.
        w (jax.Array) : initial weights of f.
        X (jax.Array) : input data to pre-train f with.
        y (jax.Array) : label data to pre-train f with.
        f (Callable) : flax CNN forward pass.
        L (Callable) : loss function.
        optimizer (Optimizer) : optimizer to train f with.
        steps (int) : number of steps to train for.
        batch_size (int) : number of examples (X, y) to use for each step.

    Returns:
        w (jax.Array) : flattened weights of CNN.

    Raises:
        None.
    """
    opt_state = optimizer.init(w)
    N = X.shape[0]

    @jax.jit
    def step(w, X, y, opt_state):
        grad = jax.grad(L)(w, X, y, f)
        updates, opt_state = optimizer.update(grad, opt_state, w)
        w = optax.apply_updates(w, updates)
        return w, opt_state

    for i in range(steps):
        inds = jr.choice(key, N, shape=(batch_size,))
        w, opt_state = step(w, X[inds], y[inds], opt_state)
        if i % 20 == 0:
            print(f"step {i} / {steps}")
        key, _ = jr.split(key)

    return w


def accuracy(w: jax.Array, X: jax.Array, y: jax.Array, f: Callable) -> jax.Array:
    """Perform forward pass with f using data (X, y) and weights w. Calculate accuracy."""
    z = f(w, X)  # forward pass with CNN
    return jnp.mean(jnp.equal(z >= 0.0, y), axis=-1)


def loss(w: jax.Array, X: jax.Array, y: jax.Array, f: Callable) -> jax.Array:
    """Perform forward pass with f using data (X, y) and weights w. Calculate BCE loss."""
    z = f(w, X)  # forward pass with CNN
    return jnp.sum(optax.sigmoid_binary_cross_entropy(z, y))


def evaluate(w: list, X: list, y: list, f: Callable) -> jax.Array:
    """Evaluate weights and data at every time step.

    Args:
        w (list) : List of weights for every filtering time step.
        X (list) : list of input test data for every filtering time step.
        y (list) : list of label test data for every filtering time step.
        f (Callable) : flax CNN forward pass.

    Returns:
        w (jax.Array) : array of accuracies at every time step.

    Raises:
        None.
    """
    acc = []
    for i in range(len(X)):
        if w[i].ndim == 2:
            accs = []
            for j in range(w[i].shape[0]):
                accs.append(accuracy(w[i][j], X[i], y[i], f))
            acc.append(accs)
        else:
            acc.append(accuracy(w[i], X[i], y[i], f))
    return jnp.array(acc)


def main(
    seed: int = 0,  # random seeds used to produce results in paper = [0 ... 9]
    steps: int = 50,
    q: float = 0.01,
    model: str = "map",  # 'pf' or 'none' or 'vkf'
    file_path: str = "file.txt",
    verbose: bool = False,
) -> None:
    # init keys
    key = jr.PRNGKey(seed)
    key, nn_key = jr.split(key)

    # init nn
    in_dim = (1, 32, 32, 1)
    f, w_init = init_nn(nn_key, in_dim)

    # init data
    X_train, y_train, X_test, y_test, X_init, y_init = Yearbook().load(key)

    # get w init by optimizing X_init, y_init for 200 steps of sgd
    if verbose:
        print("Initializing the weight matrix w...")
    optimizer = sgd(learning_rate=1e-3)
    w_init = init_w(key, w_init, X_init, y_init, f, loss, optimizer)

    # initialize filter
    if model == "map":
        filter = MAP(optimizer=adam(learning_rate=1e-3), steps=steps)
        filter.initialize(f, loss, w_init)
    elif model == "pf":
        filter = Bootstrap()
        Q = lambda x, t: q
        filter.initialize(key, Q, f, loss, w_init)
    elif model == "vkf":
        filter = VKF(optimizer=adam(learning_rate=1e-3), steps=1000)
        filter.initialize(key, q, f, w_init)
    else:
        pass  # do nothing, baseline case

    if verbose:
        print("Running " + model + " filter...")
    # run filter
    if model == "map" or model == "pf" or model == "vkf":
        w = filter.forward(X_train, y_train)
    else:
        w = jnp.tile(jnp.expand_dims(w_init, axis=0), (len(X_test), 1))  # baseline case

    # evaluate filter
    acc = evaluate(w, X_test, y_test, f)
    if verbose:
        print("Year-by-year accuracies:\n" + str(acc))

    # Use the 'a' mode to append to the file
    with open(file_path, "a") as file:
        # Convert the numpy array to a string with desired formatting
        data_str = " ".join(map(str, acc))  # Space-separated values
        # Append the string to a new line in the file
        file.write(data_str + "\n")


if __name__ == "__main__":
    typer.run(main)
