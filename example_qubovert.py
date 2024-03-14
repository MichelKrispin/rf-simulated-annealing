import qubovert as qv
import numpy as np

from timeit import default_timer as timer

from simulated_annealing_variants import (
    simulated_annealing,
    simulated_annealing_qrf,
    simulated_annealing_rf,
)
from simulated_annealing_variants.utils import f


if __name__ == "__main__":
    # Create a problem as in
    # https://qubovert.readthedocs.io/en/latest/problems/np/numberpartitioning.html
    t1 = timer()
    N = 2048
    np.random.seed(15)
    S = np.random.randint(1, 100, size=(N,)).tolist()
    # problem = qv.problems.NumberPartitioning(S)
    # H = problem.to_qubo()
    # H[()] = 0  # Remove constant
    # Q = qv.utils.qubo_to_matrix(H)
    c = np.sum(S)
    Q = np.zeros((N, N))
    for i in range(N):
        Q[i, i] = S[i] * (S[i] - c)
        for j in range(i + 1, N):
            Q[i, j] = 2 * S[i] * S[j]
    print(f"Formulation time for N = {N} is {timer() - t1:.4f}s")

    # t1 = timer()
    # res = qv.sim.anneal_qubo(H, num_anneals=10, seed=34)
    # t2 = timer()
    # x = np.array([res.best.state[i] for i in range(N)])
    # energy = f(x, Q)
    # print(f"Qubovert annealing:   {energy} ({t2 - t1:.4f}s)")

    t1 = timer()
    x, _ = simulated_annealing(Q, 10000)
    t2 = timer()
    energy = f(x, Q)
    print(f"Default annealing:    {energy} ({t2 - t1:.4f}s)")

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
