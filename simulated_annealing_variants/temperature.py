import numpy as np

from typing import Tuple
from enum import Enum


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


def positive_expectation_delta_x(Q: np.ndarray, m: int):
    """Compute the expected energy difference of a **worse** bit flip.
    In contrast to the expectation delta, this function averages
    only worse flips.

    Args:
        Q (np.ndarray): The QUBO matrix.
        m (int): The number of samples taken.

    Returns:
        float: The expected worse energy difference of a bit flip.
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
        # Figure out positive elements of h and count them
        p = h[h > 0]
        r += np.sum(p) / p.shape[0]
    return r / m


def iterative_temperature(Q: np.ndarray, p_start: float = 0.5) -> Tuple[float, float]:
    """Sample the start temperature iterative following Walid Ben-Ameur at
    https://link.springer.com/article/10.1023/B:COAP.0000044187.23143.bd

    Args:
        Q (np.ndarray): The QUBO matrix as an upper triangular matrix.
        p_start (float, optional): The flip probability of a worse solution at the start. Defaults to 0.99.

    Returns:
        Tuple[float, float]: The start and end temperature given as a tuple.
    """
    raise NotImplementedError("A small todo here...")


def sample_temperature(
    Q: np.ndarray,
    p_start: float = 0.99,
    p_trans: float = 0.5,
    nu: float = 0.99,
) -> Tuple[float, float]:
    """Sample the start and end values of the simulated annealing temperature schedule.

    Args:
        Q (np.ndarray): The QUBO matrix.
        p_start (float, optional): Flip probability at the beginning. Defaults to 0.99.
        p_trans (float, optional): Flip probability after update step. Defaults to 0.5.
        nu (float, optional): End temperature parameter. Defaults to 0.99.

    Returns:
        Tuple[float, float]: Start temperature, end temperature.
    """
    N = Q.shape[0]
    e_delta_x = 2 * np.abs(expectation_delta_x(Q, 8))
    t_0 = -e_delta_x / (np.log(1 - np.power(1 - p_start, 1.0 / N)))
    t_end = np.power(t_0, 1 - (1.0 / nu)) * np.power(
        -e_delta_x / (np.log(1 - np.power(1 - p_trans, 1.0 / N))),
        -1.0 / nu,
    )
    return t_0, t_end


def deterministic_temperature(
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

    t_0 = -max_del_energy / np.log(start_flip_prob)
    t_end = -min_del_energy / np.log(end_flip_prob)
    return t_0, t_end


def geometric_temperature_schedule(
    t_0: float, t_end: float, num_t_values: int, generate_inverse: bool = True
) -> np.ndarray:
    """Compute the complete simulated annealing temperature schedule using the parameters.
    Computes the beta schedule, i.e., the inverse temperature values, by default.

    Args:
        t_0 (float): Start temperature.
        t_end (float): End temperature.
        num_t_values (int): Number of iterations.
        generate_inverse (bool, optional): If the inverse (beta schedule) should be computed. Defaults to True.

    Returns:
        np.ndarray: The temperature values.
    """
    epsilon = np.exp(np.log(t_end / t_0) / num_t_values)
    ts = np.zeros(num_t_values)
    for i in range(0, num_t_values):
        ts[i] = t_0 * epsilon**i

    if not generate_inverse:
        return ts

    # Otherwise invert the temperature
    for i in range(len(ts)):
        ts[i] = 1.0 / ts[i]
    return ts


TEMPERATURE_SAMPLING_MODE = Enum(
    "TemperatureSampling", ["deterministic", "sample", "iterative"]
)
COOLING_TYPE = Enum("Cooling", ["geometric"])


def temperature_schedule(
    Q: np.ndarray,
    num_t_values: int,
    temperature_sampling_mode: Enum = TEMPERATURE_SAMPLING_MODE.deterministic,
    cooling_type: Enum = COOLING_TYPE.geometric,
    generate_inverse: bool = True,
) -> np.ndarray:
    """Create a cooling schedule for simulated annealing.

    Args:
        Q (np.ndarray): The QUBO matrix.
        num_t_values (int): The number of iterations.
        temperature_sampling_mode (Enum, optional): How to sample the start and end temperature. Defaults to TEMPERATURE_SAMPLING_TYPE.deterministic.
        cooling_type (Enum, optional): The cooling schedule decay. Defaults to COOLING_TYPE.geometric.
        generate_inverse (bool, optional): Create a beta schedule, i.e., 1/t_i. Defaults to True.

    Raises:
        TypeError: For unknown temperature sampling modes.
        TypeError: For unkown cooling schedules.

    Returns:
        np.ndarray: The temperature values as a list.
    """
    if temperature_sampling_mode == TEMPERATURE_SAMPLING_MODE.deterministic:
        t_0, t_end = deterministic_temperature(Q)
    elif temperature_sampling_mode == TEMPERATURE_SAMPLING_MODE.sample:
        t_0, t_end = sample_temperature(Q)
    elif temperature_sampling_mode == TEMPERATURE_SAMPLING_MODE.iterative:
        t_0, t_end = iterative_temperature(Q)
    else:
        raise TypeError(
            f"Temperature sampling mode {temperature_sampling_mode} is not implemented"
        )

    if cooling_type == COOLING_TYPE.geometric:
        return geometric_temperature_schedule(
            t_0, t_end, num_t_values, generate_inverse
        )
    else:
        raise TypeError(f"Cooling type {cooling_type} is not implemented")
