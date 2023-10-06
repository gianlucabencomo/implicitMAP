import numpy as np
from numpy.linalg import inv, cholesky
from scipy.linalg import schur

import matplotlib.pyplot as plt


def symmetric_diag(A, B):
    """Symmetric diagonalization from Santos (1996) for converting between explicit priors and gradient descent."""
    G = cholesky(A)
    C = inv(G) @ B @ inv(G.T)

    # https://math.stackexchange.com/questions/875729/decompose-a-real-symmetric-matrix
    T, Z = schur(C, output="complex")
    T = np.real(T)
    Z = np.real(Z)

    X = inv(G.T) @ Z
    a = np.diag(Z.T @ C @ Z)

    return X, a


def RMSE(y: np.array, yhat: np.array):
    """Root mean square error."""
    return ((y - yhat) ** 2.0).mean() ** (1.0 / 2.0)


def plot1d(x, label=None, alpha=None, c=None):
    """Plotting function for 1-D case."""
    if label == "True State" and alpha is None:
        plt.plot(x, "ko", label=label, markersize=4.2)
    elif alpha is None:
        plt.plot(x, "-", c=c, label=label, linewidth=2.5)
    elif alpha is not None:
        plt.plot(x, "-", c=c, alpha=alpha, zorder=1, label=label, linewidth=2.5)
    legend = plt.legend(
        loc="upper center",
        fontsize=13,
        ncols=5,
        frameon=False,
        bbox_to_anchor=(0.5, 1.15),
    )
    for handle in legend.legendHandles:
        handle.set_linewidth(3.5)  # Optional: Adjust the handle line width
    plt.xlabel("Timestep", fontsize=13, fontweight="bold")
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)


def plothist(pf_x, map_x, nrows=10, ncols=10, x_true=None):
    """Plot the MAP estimates over the particle filter filtering distributions."""
    # Create a 4x5 subplot grid
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols, nrows))
    c = int(200 / (nrows * ncols))
    inds = np.arange(0, 200, c)

    # Iterate through the subplots and plot histograms
    for i in range(nrows):
        for j in range(ncols):
            ax = axes[i, j]
            ax.hist(
                pf_x[inds[i * nrows + j]],
                bins=100,
                density=True,
                alpha=0.5,
                color="blue",
            )  # Adjust the number of bins as needed
            ax.axvline(x=map_x[inds[i * nrows + j]], color="red", linestyle="--")
            if x_true is not None:
                ax.axvline(x=x_true[inds[i * nrows + j]], color="black", linestyle="--")
            ax.set_title(f"T = {inds[i * nrows + j]}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

    # Adjust layout and show the plot
    plt.tight_layout()
