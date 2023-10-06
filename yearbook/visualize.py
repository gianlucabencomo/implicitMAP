import typer
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-paper")

BASELINE = "res/baseline.txt"
DIRECT = "res/adam1000.txt"
VKF = [
    "res/vkf0_1.txt",
    "res/vkf0_05.txt",
    "res/vkf0_01.txt",
    "res/vkf0_005.txt",
    "res/vkf0_001.txt",
]
PF = [
    "res/pf0_1.txt",
    "res/pf0_05.txt",
    "res/pf0_01.txt",
    "res/pf0_005.txt",
    "res/pf0_001.txt",
]
MAP = [
    "res/adam1.txt",
    "res/adam10.txt",
    "res/adam25.txt",
    "res/adam50.txt",
    "res/adam100.txt",
]


def plot(method: list, verbose: bool) -> None:
    """Plots and prints results reported in paper for yearbook dataset.

    Args:
        method (list) : file names to load
        verbose (bool) : print out results

    Returns:
        None.

    Raises:
        None.
    """
    plt.figure(figsize=(14, 5))
    # plot stationary baseline
    with open(BASELINE, "r") as file:
        lines = file.readlines()
    stationary = []
    for line in lines:
        accs = line.strip().split(" ")
        for i in range(len(accs)):
            accs[i] = float(accs[i])
        stationary.append(accs)
    stationary = np.array(stationary)
    mu = 100 * stationary.mean(axis=0)
    years = np.arange(1931, mu.shape[0] + 1931)
    plt.plot(years, mu, label="Static Weights", linewidth=2.5)
    if verbose:
        ci = 1.96 * mu[:40].std() / np.sqrt(mu[:40].shape[0])
        print(f"Baseline (val) = {mu[:40].mean():.3f} \pm {ci:.3f}")
        ci = 1.96 * mu[40:].std() / np.sqrt(mu[40:].shape[0])
        print(f"Baseline (test) = {mu[40:].mean():.3f} \pm {ci:.3f}")

    if method[0] == "res/adam1.txt":
        # if method is MAP, plot direct fit
        with open(DIRECT, "r") as file:
            lines = file.readlines()

        direct = []
        for line in lines:
            accs = line.strip().split(" ")
            for i in range(len(accs)):
                accs[i] = float(accs[i])
            direct.append(accs)
        direct = np.array(direct)
        mu = 100 * direct.mean(axis=0)
        plt.plot(years, mu, label="Direct Fit", linewidth=2.5)
        if verbose:
            ci = 1.96 * mu[:40].std() / np.sqrt(mu[:40].shape[0])
            print(f"Direct Fit (val) = {mu[:40].mean():.3f} \pm {ci:.3f}")
            ci = 1.96 * mu[40:].std() / np.sqrt(mu[40:].shape[0])
            print(f"Direct Fit (test) = {mu[40:].mean():.3f} \pm {ci:.3f}")

    for f in method:
        # plot values for each method
        with open(f, "r") as file:
            lines = file.readlines()
        res = []
        for line in lines:
            accs = line.strip().split(" ")
            for i in range(len(accs)):
                accs[i] = float(accs[i])
            res.append(accs)
        res = np.array(res)
        mu = 100 * res.mean(axis=0)
        plt.plot(years, mu, label=f[4:-4], linewidth=2.5)
        if verbose:
            ci = 1.96 * mu[:40].std() / np.sqrt(mu[:40].shape[0])
            print(f[4:-4] + f" (val) = {mu[:40].mean():.3f} \pm {ci:.3f}")
            ci = 1.96 * mu[40:].std() / np.sqrt(mu[40:].shape[0])
            print(f[4:-4] + f" (test) = {mu[40:].mean():.3f} \pm {ci:.3f}")

    legend = plt.legend(
        loc="upper center",
        fontsize=15,
        ncols=6,
        frameon=False,
        bbox_to_anchor=(0.5, 1.15),
    )
    for handle in legend.legendHandles:
        handle.set_linewidth(3.5)  # Optional: Adjust the handle line width
    plt.xlabel("Year", fontsize=15, fontweight="bold")
    plt.ylabel("Classification Accruacy (%)", fontsize=15, fontweight="bold")
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(True)


def main(verbose: bool = False) -> None:
    # plot three methods
    plot(VKF, verbose)
    plot(PF, verbose)
    plot(MAP, verbose)
    plt.show()


if __name__ == "__main__":
    typer.run(main)
