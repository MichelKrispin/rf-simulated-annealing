import numpy as np

from timeit import default_timer as timer

from simulated_annealing_variants import (
    simulated_annealing,
    simulated_annealing_qrf,
    simulated_annealing_rf,
)
from simulated_annealing_variants.utils import f
from simulated_annealing_variants.temperature import TEMPERATURE_SAMPLING_MODE


if __name__ == "__main__":
    # Create a random graph with an edge connection probability of 0.5
    # and formulate a max clique problem.

    t1 = timer()
    N = 2048
    np.random.seed(15)
    A = np.triu(np.random.randint(0, 2, (N, N)))
    A = A.T + A
    Q = 2 * (1 - A)
    np.fill_diagonal(Q, -1)

    print(f"Formulation time for N = {N} is {timer() - t1:.4f}s")

    t1 = timer()
    x, _ = simulated_annealing(Q, 100000)
    t2 = timer()
    energy = f(x, Q)
    print(f"Default annealing (d): {energy} ({t2 - t1:.4f}s)")

    t1 = timer()
    x, _ = simulated_annealing_qrf(Q, 10000)
    t2 = timer()
    energy = f(x, Q)
    print(f"Quasi rejection-free: {energy} ({t2 - t1:.4f}s)")

    t1 = timer()
    x, _ = simulated_annealing_rf(Q, 10000)
    t2 = timer()
    energy = f(x, Q)
    print(f"Rejection-free:       {energy} ({t2 - t1:.4f}s)")
