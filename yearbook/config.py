from typing import Tuple


def get_config() -> Tuple[list, list, list]:
    steps = [1, 3, 5, 10, 25, 50, 100]
    decay = [0.1, 0.3, 0.5, 0.7, 0.9]
    alpha = [1e-1, 1e-2, 1e-3, 1e-4]
    return steps, decay, alpha
