import numpy as np


def f(x: np.ndarray, Q: np.ndarray) -> float:
    """The QUBO function for one time energy computation.

    Args:
        x (np.ndarray): The x vector.
        Q (np.ndarray): The QUBO matrix.

    Returns:
        float: The energy.
    """
    return x.T @ Q @ x
