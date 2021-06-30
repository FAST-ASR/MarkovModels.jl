# MarkovModels

*Julia package for inference with (Hidden) Markov Models with CPU or
GPU.*

| **Documentation**  |
|:------------------:|
|[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://lucasondel.github.io/MarkovModels.jl/stable) [![](https://img.shields.io/badge/docs-dev-blue.svg)](https://lucasondel.github.io/MarkovModels.jl/dev)|

## Installation

The package can be installed with the Julia package manager.
From the Julia REPL, type ] to enter the Pkg REPL mode and run:

```julia
pkg> add MarkovModels
```

Have a look at the [documentation](https://lucasondel.github.io/MarkovModels.jl/stable)
or the [examples](https://github.com/lucasondel/MarkovModels.jl/tree/master/examples) 
to get started.

## Benchmark

Here is a small benchmark of the *forward-backward* algorithm run
with a Intel(R) Xeon(R) CPU E5-2630 v2 @ 2.60GHz and a GeForce GTX 1080
GPU.

The computation load is approximately of aligning 10 seconds of
speech with 30 phones each with 3-states left-to-right topology.

```
$ julia --project examples/benchmark.jl -N 1000 -S 300
Setup:
  float type: Float64
  # states: 300
  # frames: 1000
  # pruning: Inf

αβrecursion with dense CPU arrays:
  5.282 s (2015 allocations: 14.05 MiB)
------------------------------------------------
αβrecursion with sparse CPU arrays:
  3.051 s (119692 allocations: 3.34 GiB)
------------------------------------------------
αβrecursion with dense GPU arrays:
  1.329 s (322411 allocations: 8.28 MiB)
------------------------------------------------
αβrecursion with sparse GPU arrays:
  50.945 ms (84629 allocations: 3.92 MiB)
------------------------------------------------
```

## Authors

* [Lucas Ondel](https://lucasondel.github.io/), Brno University of Technology, 2020, [Laboratoire Interdisciplinaire des Sciences du Numérique](https://www.lisn.upsaclay.fr/) 2021
* Martin Kocour, Brno University of Technology, 2020

