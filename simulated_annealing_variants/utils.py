import numpy as np
from typing import Tuple


def expectation_delta_x(Q: np.ndarray, m: int):
    """Compute the expected energy difference of a bit flip.

    Args:
        Q (np.ndarray): The QUBO matrix.
        m (int): The number of samples taken.

    Returns:
        float: The expected energy difference of a bit flip.
    """
    N = Q.shape[0]
    Q_diag = np.diag(Q)
    Q_full = Q + Q.T
    np.fill_diagonal(Q_full, Q_diag)

    flips = np.random.randint(0, N, (m,))
    x = np.random.randint(0, 2, (N,))
    h = np.sum(Q_full * x, axis=1) + (1 - x) * Q_diag

    r = np.sum(h) / N
    for i in range(m):
        dh = Q_full[flips[i]] * (1 - 2 * x[flips[i]])
        x[flips[i]] = 1 - x[flips[i]]
        dh[flips[i]] = 0
        h -= dh
        r += np.sum(np.abs(h)) / N
    return r / m


def sample_temperature(
    Q: np.ndarray,
    p_start: float = 0.99,
    p_trans: float = 0.5,
    nu: float = 0.99,
    k: int = 10,
) -> Tuple[float, float, float]:
    """Sample the start and end values of the simulated annealing temperature schedule.

    Args:
        Q (np.ndarray): The QUBO matrix.
        p_start (float, optional): Flip probability at the beginning. Defaults to 0.99.
        p_trans (float, optional): Flip probability after update step. Defaults to 0.5.
        nu (float, optional): End temperature parameter. Defaults to 0.99.
        k (int, optional): After how many steps the rim of a local minimum is expected to be reached. Defaults to 10.

    Returns:
        Tuple[float, float, float]: Start temperature, end temperature, local minimum offset rate.
    """
    N = Q.shape[0]
    e_delta_x = 2 * np.abs(expectation_delta_x(Q, 8))
    t_0 = -e_delta_x / (np.log(1 - np.power(1 - p_start, 1.0 / N)))
    t_end = np.power(t_0, 1 - (1.0 / nu)) * np.power(
        -e_delta_x / (np.log(1 - np.power(1 - p_trans, 1.0 / N))),
        -1.0 / nu,
    )
    offset_increase_rate = e_delta_x / k
    return t_0, t_end, offset_increase_rate


def sample_temperature_range(
    Q: np.ndarray,
    start_flip_prob: float = 0.5,
    end_flip_prob: float = 0.01,
    factor: float = 2.0,
) -> Tuple[float, float]:
    """Sample the temperature range for the annealing process.
    Pretty much exactly from https://github.com/jtiosue/qubovert/blob/master/qubovert/sim/_anneal_temperature_range.py

    Args:
        Q (np.ndarray): The QUBO matrix.
        start_flip_prob (float): Flip probability at the start.
        end_flip_prob (float): Flip probability at the end.
        factor (float): Factor for approximating the maximum/minimum energy difference.

    Returns:
        Tuple[float, float]: The start and end time as a tuple.
    """
    min_del_energy = factor * np.min(np.abs(Q)[np.nonzero(Q)])
    Q_full = Q.T + Q
    np.fill_diagonal(Q_full, np.diagonal(Q))
    max_del_energy = factor * np.max(np.sum(np.abs(Q_full), axis=0))

    t0 = -max_del_energy / np.log(start_flip_prob)
    t_end = -min_del_energy / np.log(end_flip_prob)
    return t0, t_end


def temperature_schedule(
    t0: float, t_end: float, num_t_values: int, generate_inverse: bool = True
) -> np.ndarray:
    """Compute the complete simulated annealing temperature schedule using the parameters.
    Computes the beta schedule, i.e., the inverse temperature values, by default.

    Args:
        t0 (float): Start temperature.
        t_end (float): End temperature.
        num_t_values (int): Number of iterations.
        generate_inverse (bool, optional): If the inverse (beta schedule) should be computed. Defaults to True.

    Returns:
        np.ndarray: The temperature values.
    """
    epsilon = np.exp(np.log(t_end / t0) / num_t_values)
    ts = np.zeros(num_t_values)
    for i in range(0, num_t_values):
        ts[i] = t0 * epsilon**i

    if not generate_inverse:
        return ts

    # Otherwise invert the temperature
    for i in range(len(ts)):
        ts[i] = 1.0 / ts[i]
    return ts


def f(x: np.ndarray, Q: np.ndarray) -> float:
    """The QUBO function for one time energy computation.

    Args:
        x (np.ndarray): The x vector.
        Q (np.ndarray): The QUBO matrix.

    Returns:
        float: The energy.
    """
    return x.T @ Q @ x
