import numpy as np
from typing import Tuple

from .utils import f
from .temperature import temperature_schedule, TEMPERATURE_SAMPLING_MODE


def simulated_annealing(
    Q: np.ndarray,
    num_t_values: int | None = None,
    temperature_sampling_mode: TEMPERATURE_SAMPLING_MODE = TEMPERATURE_SAMPLING_MODE.deterministic,
    seed: int | None = None,
) -> Tuple[np.ndarray, float]:
    """Simulated annealing with a computational complexity of O(n * t),
    where t is the number of timesteps.
    This is achieved by computing only the updated values which are at most
    n per update step.

    Args:
        Q (np.ndarray): The QUBO matrix.
        num_t_values (int | None, optional): Number of update steps. Defaults to the size of the QUBO squared.
        temperature_sampling_mode (TEMPERATURE_SAMPLING_TYPE): The way of sampling the temperature start and end values. Defaults to deterministic.
        seed (int | None, optional): Random seed. Defaults to None.


    Returns:
        Tuple[np.ndarray, float]: The best solutions and its energy.
    """
    rng = np.random.Generator(np.random.PCG64(seed=seed))

    # Create helper matrix
    n = Q.shape[0]
    Q_outer = Q + Q.T
    np.fill_diagonal(Q_outer, 0)

    if num_t_values is None:
        num_t_values = n**2

    # Random initial
    x = rng.integers(0, high=2, size=(n,))
    f_x = f(x, Q)

    # Create the inverted temperature values
    betas = temperature_schedule(
        Q,
        num_t_values=num_t_values,
        temperature_sampling_mode=temperature_sampling_mode,
        generate_inverse=True,
    )

    for beta in betas:
        # Random flip in x
        idx = rng.integers(0, high=n)

        # Compute the difference between the flip and the previous energy
        sign = -(2 * x[idx] - 1)
        f_difference = sign * (np.dot(x, Q_outer[idx]) + Q[idx, idx])
        f_y = f_x + f_difference

        # Accept the new one if better (t is inverted beforehand)
        if f_y <= f_x or (np.exp(-(f_y - f_x) * beta) > rng.uniform(0, 1)):
            x[idx] = 1 - x[idx]
            f_x = f_y

    return x, f_x


def simulated_annealing_slow(
    Q: np.ndarray, num_t_values: int | None = None, seed: int | None = None
) -> Tuple[np.ndarray, float]:
    """Classical simulated annealing with a computational complexity of O(n^2 * t),
    where t is the number of timesteps.
    This is achieved by computing only the updated values which are at most
    n per update step.

    Args:
        Q (np.ndarray): The QUBO matrix.
        num_t_values (int | None, optional): Number of update steps. Defaults to the size of the QUBO squared.
        seed (int | None, optional): Random seed. Defaults to None.

    Returns:
        Tuple[np.ndarray, float]: The best solutions and its energy.
    """
    rng = np.random.Generator(np.random.PCG64(seed=seed))
    n = Q.shape[0]

    if num_t_values is None:
        num_t_values = n**2

    # Create the beta schedule.
    betas = temperature_schedule(Q, num_t_values=num_t_values, generate_inverse=True)

    # Random initial x
    x = rng.integers(0, high=2, size=(n,))
    f_x = f(x, Q)

    for beta in betas:
        # Random flip in x
        idx = np.random.randint(n)
        x[idx] = 1 - x[idx]

        # Compute differences
        f_y = f(x, Q)

        # Accept the new one if better
        if f_y <= f_x or (np.exp(-(f_y - f_x) * beta) > np.random.uniform()):
            f_x = f_y
        else:
            # Otherwise flip back
            x[idx] = 1 - x[idx]

    return x, f_x
