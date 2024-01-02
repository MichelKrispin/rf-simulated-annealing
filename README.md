# (Quasi) Rejection-Free Simulated Annealing

Python implementations of quasi rejection-free and rejection-free simulated annealing to optimize [QUBO](https://en.wikipedia.org/wiki/Quadratic_unconstrained_binary_optimization) problems.

The only requirement for the algorithms is `numpy` while the example uses [qubovert](https://github.com/jtiosue/qubovert) for the QUBO formulation.

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
