from mpi4py import MPI
from simulated_annealing_variants import simulated_annealing
import numpy as np
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# =========
# Load some qubo matrix
np.random.seed(15)
Q = np.triu(np.random.uniform(-12, 22, size=(8192, 8192)))
# =========


num_iterations = 25000

# Run the simulated annealing method in each instance
local_seed = int(time.time() * 1000) % 10000
x, best_energy = simulated_annealing(Q, num_iterations, seed=local_seed)
print(f"[{rank}]: {best_energy}")

# Get the best result of all instances
best_energy, best_rank = comm.allreduce((best_energy, rank), op=MPI.MINLOC)


# Then send the best one to the root
if best_rank != 0 and rank == best_rank:
    comm.Send(x, dest=0)
if rank == 0:
    if best_rank != 0:
        best_x = np.empty(Q.shape[0], dtype=np.float64)
        comm.Recv(best_x, source=best_rank)
    else:
        best_x = x

    print(
        f"\n\n   ---\n{best_rank} found best energy: {best_energy}"
    )  #  and \nx: {best_x}")

    # Use the best_x here...
