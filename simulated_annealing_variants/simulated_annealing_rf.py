import numpy as np
from typing import Tuple

from .utils import sample_temperature, temperature_schedule, f


def simulated_annealing_rf(
    Q: np.ndarray, num_t_values: int, seed: int | None = None
) -> Tuple[np.ndarray, float]:
    """Rejection-free simulated annealing with parallelized update scheme.

    Args:
        Q (np.ndarray): The QUBO matrix.
        num_t_values (int): Number of update steps.

    Returns:
        Tuple[np.ndarray, float]: The best solutions and its energy.
    """
    rng = np.random.Generator(np.random.PCG64(seed=seed))

    # Number of bits
    n = Q.shape[0]

    # For easier computation create a dense matrix
    Q_diag = np.diag(Q)
    Q_full = Q + Q.T
    np.fill_diagonal(Q_full, Q_diag)

    t0, t_end, _ = sample_temperature(Q)  # Sample randomly

    # Create the inverted temperature values
    ts = temperature_schedule(
        t0=t0, t_end=t_end, num_t_values=num_t_values, generate_inverse=False
    )

    # Random initial x
    x = rng.integers(0, high=2, size=(n,))

    # Remember best values
    best_x = np.copy(x)
    f_x = f(x, Q)
    best_energy = f_x

    # ---------------- Start

    # The change of delta E with respect to a bitflip at index [i]
    # Initial flip
    h = np.sum(Q_full * x, axis=1) + (1 - x) * Q_diag

    for t in ts:
        # Compute the differene for all flipped x at once
        delta_E = -(1 - 2 * (1 - x)) * h

        # Compute criteria
        u_s = rng.uniform(0, 1, size=delta_E.shape)
        criteria = np.maximum(0, delta_E) + t * np.log(-np.log(u_s))
        accepted_state_idx = criteria.argmin()

        # Accept the state by flipping x
        x[accepted_state_idx] = 1 - x[accepted_state_idx]

        # Check for best solution
        f_x += delta_E[accepted_state_idx]
        if f_x < best_energy:
            best_x = np.copy(x)
            best_energy = f_x

        # Then update the h
        dh = Q_full[accepted_state_idx] * (1 - 2 * x[accepted_state_idx])
        dh[accepted_state_idx] = 0
        h -= dh

    return best_x, best_energy
