# Inference

## Baum-Welch algoritm (forward-backward)

The Baum-Welch algorithm computes the probability to be in a state `i`
at time $n$:
```math
p(z_n = i | x_1, ..., x_N)
```
It is implemented by the [`αβrecursion`](@ref) function.

```@docs
αβrecursion
```
