# (Quasi) Rejection-Free Simulated Annealing

Python implementations of quasi rejection-free and rejection-free simulated annealing to optimize [QUBO](https://en.wikipedia.org/wiki/Quadratic_unconstrained_binary_optimization) problems.
A [CUDA implementation](https://github.com/MichelKrispin/cuda-rf-simulated-annealing) exists as well.
Additionally, the default simulated annealing algorithm is implemented as a reference.
This reference implementation makes use of an improved update scheme where only the differences to the last energy state is computed.
By using this scheme, the complexity is reduced to O(n \* t), where n is the number of bits in the problem and t is the number of time steps.

## Usage

The package is available at [pypi](https://pypi.org/project/simulated-annealing-variants/), so it can be installed with `pip install simulated-annealing-variants`.
The only requirement for the algorithms is `numpy` and they accept an upper triangular matrix while the example uses [qubovert](https://github.com/jtiosue/qubovert) for the QUBO formulation.
The algorithm is simply invoked by

```python
from simulated_annealing_variants import simulated_annealing, simulated_annealing_qrf, simulated_annealing_rf
x, energy = simulated_annealing(Q=Q, num_t_values=10000)
x, energy = simulated_annealing_qrf(Q=Q, num_t_values=10000)
x, energy = simulated_annealing_rf(Q=Q, num_t_values=10000)
```

## Algorithm

The implementation uses a parallel computation scheme and has therefore a quadratic speedup compared to standard [simulated annealing implementations](https://en.wikipedia.org/wiki/Simulated_annealing#Pseudocode).
The following pseudocode describes the idea and the difference to the standard implementation.
For the exact implementation of the parallel computation have a look at the [quasi rejection-free](./simulated_annealing/simulated_annealing_qrf.py) or the [rejection-free](./simulated_annealing/simulated_annealing_rf.py) python file.

```
Procedure RFSimulatedAnnealing
    x = random initial state
    xb = x
    For i in 0:N
        x = FindLocalNeighbour(f, x, T[i])
        If f(x) < f(xb)
            xb = x
        EndIf
    EndFor
    return xb
EndProcedure

Procedure RFFindLocalNeighbour(f, x, t)
    DeltaE = empty vector
    u = empty vector
    For i in 1:length(x)
        DelatE[i] = f(x) - f(flipat(x, i)) # Energy difference for x flipped at bit i
        u[i] = random(0, 1)
    EndFor
    accepted_idx = argmin_i {max(0, DeltaE[i]) + t log( -log(u[i])) }
    return flipat(x, accepted_idx)
EndProcedure
```

## MPI

The MPI version makes use of [mpi4py](https://mpi4py.readthedocs.io/en/stable/index.html).
So, after installing it with `pip install mpi4py` the parallel version can be run with `mpiexec -n 4 python mpi_simulated_annealing.py`.
Currently, each process computes an individual solution, prints the found energy and sends the corresponding x vector to the root process.
The [file](./mpi_simulated_annealing.py) can be altered to make use of the best solution vector as well as changing the used QUBO problem.
