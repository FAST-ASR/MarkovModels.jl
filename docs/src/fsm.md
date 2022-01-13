# Finite State Machines

This package represents Markov chains as probabilistic a Finite State
Machine (FSM).  Here is an example of FSM as used by the package:

![missing image](images/examplefsm.svg)

The thick line node indicates the starting state whereas the double
line node indicates the ending state.

!!! note
    To be able to visualize FSMs as in the example above when using
    [IJulia](https://github.com/JuliaLang/IJulia.jl) or [Pluto](https://github.com/fonsp/Pluto.jl),
    make sure that the `dot` program (from [graphviz](https://graphviz.org/))
    is available in your shell `PATH` variable. Also, you won't be able
    to visualize the FSM in the REPL.

Note that, contrary to standard FSM, our implementation puts the labels
in the states rather than on the arcs. This is equivalent of a
constrained [Weighted Finite State Acceptor](https://en.wikipedia.org/wiki/Finite-state_transducer)
where all the incoming arcs of a given states share the same labels.
We have made this choice to facilitate the interpretation of our FSMs
as Markov chains.

In the following, we present the tools provided by the package
to manipulate such FSM. All the examples below assume that you
have already imported the MarkovModels.jl package by doing `using
MarkovModels`.

## FSM interface

All the FSMs are subtypes from the following abstract type:
```@docs
AbstractFSM
```
They support the following functions:
```@docs
states
arcs
Base.length(::AbstractFSM)
```

## Mutable FSM interface

FSM that can be changed in place (e.g. adding states/arcs) are
subtypes of the following abstract type:
```@docs
AbstractMutableFSM
```
They support the following functions:
```@docs
addstate!
addarc!
```

## Concrete FSMs

The package has the following concrete FSM types:
```@docs
VectorFSM
HierarchicalFSM
MatrixFSM
```
## FSM operations

```@docs
Base.union(::AbstractFSM{T}, ::AbstractFSM{T}) where T
determinize
minimize
renormalize
transpose
```
