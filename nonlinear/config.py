from typing import Tuple


def get_config(system: str) -> Tuple[list, list, list, list, list, list]:
    """Returns configurations to do a grid search over for each optimizer."""
    steps = [1, 3, 5, 10, 25, 50, 100]
    # learning rates terms for adagrad
    adagrad_hyp = [
        1.0,
        5e-1,
        1e-1,
        5e-2,
        1e-2,
    ]
    # learning rates for gradient descent
    gd_hyp = [
        1.0,
        5e-1,
        1e-1,
        5e-2,
        1e-2,
    ]
    # learning rate and decay terms for rmsprop
    rmsprop_hyp = [
        (1.0, 0.9),
        (5e-1, 0.9),
        (1e-1, 0.9),
        (5e-2, 0.9),
        (1e-2, 0.9),
        (1.0, 0.5),
        (5e-1, 0.5),
        (1e-1, 0.5),
        (5e-2, 0.5),
        (1e-2, 0.5),
        (1.0, 0.1),
        (5e-1, 0.1),
        (1e-1, 0.1),
        (5e-2, 0.1),
        (1e-2, 0.1),
    ]
    # learning rate and decay terms for adam
    adam_hyp = [
        (1.0, 0.9, 0.9),
        (5e-1, 0.9, 0.9),
        (1e-1, 0.9, 0.9),
        (5e-2, 0.9, 0.9),
        (1e-2, 0.9, 0.9),
        (1.0, 0.5, 0.5),
        (5e-1, 0.5, 0.5),
        (1e-1, 0.5, 0.5),
        (5e-2, 0.5, 0.5),
        (1e-2, 0.5, 0.5),
        (1.0, 0.1, 0.1),
        (5e-1, 0.1, 0.1),
        (1e-1, 0.1, 0.1),
        (5e-2, 0.1, 0.1),
        (1e-2, 0.1, 0.1),
    ]
    # don't run AMS
    ams_hyp = []
    return steps, adagrad_hyp, adam_hyp, rmsprop_hyp, ams_hyp, gd_hyp
