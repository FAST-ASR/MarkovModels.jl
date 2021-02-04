# MarkovModels - Different pruning strategies.
#
# Lucas Ondel, 2020
#

abstract type PruningStrategy end

#######################################################################
# Default pruning stragegy: do nothing

struct NoPruning <: PruningStrategy end
const nopruning = NoPruning()
(::NoPruning)(candidates, n, N) = candidates

#######################################################################
# Safe pruning

"""
    struct ThresholdPruning
        ...
    end

Prune the states that cannot reach the final state before the end of
the sequence.

# Constructor
```julia
SafePruning(fsm)
```
"""
struct SafePruning <: PruningStrategy
    distances::Dict
end

function SafePruning(fsm::AbstractFSM)
    fsmᵀ = fsm |> transpose
    distances = Dict()
    visited = Set()
    stack = [(initstate(fsmᵀ), 0)]
    while ! isempty(stack)
        state, steps = popfirst!(stack)

        for (nstate, _) in nextemittingstates(state)
            cdist = get(distances, nstate, typemax(typeof(steps)))
            distances[nstate] = min(cdist, steps + 1)
            if nstate ∉ visited
                push!(visited, nstate)
                push!(stack, (nstate, steps + 1))
            end
        end
    end
    SafePruning(distances)
end

function (pruning::SafePruning)(candidates, n, N)
    function filt(p)
        if isfinal(p.first) return true end
        dist = get(pruning.distances, p.first, 1)
        if dist - 1 ≤ N-n return true end
        return false
    end
    filter!(filt, candidates)
end

#######################################################################
# Threshold pruning

"""
    struct ThresholdPruning
        ...
    end

Prune the active states (tokens) that have weights lower than the
maximum weight at a given time frame minus a threshold `Δ`. The lower
the threshold `Δ` the more the pruning:
  * when `Δ < 0` all paths are discarded, don't use negative threshold !!
  * `Δ == 0` greedy decoding, only keep the most likely state at each
    time step
  * `Δ = +∞` no pruning

# Constructor
```julia
ThresholdPruning(Δ)
```
"""
struct ThresholdPruning <: PruningStrategy
    Δ::Float64
end

function (pruning::ThresholdPruning)(candidates, n, N)
    maxval = maximum(p -> p.second, candidates)
    filter!(p -> maxval - p.second ≤ pruning.Δ, candidates)
end

#######################################################################
# Special backward pruning

struct BackwardPruning <: PruningStrategy
    lnα::Vector{Dict{State,<:AbstractFloat}}
end

function (pruning::BackwardPruning)(candidates, n, N)
    filter!(p -> p.first ∈ keys(pruning.lnα[n]), candidates)
end

#######################################################################
# Composition of pruning strategies

struct CompoundPruning{T1<:PruningStrategy,T2<:PruningStrategy} <: PruningStrategy
    strat1::T1
    strat2::T2
end

Base.:∘(s1::PruningStrategy, s2::PruningStrategy) = CompoundPruning(s1, s2)

function (pruning::CompoundPruning)(candidates, n, N)
    pruning.strat1(pruning.strat2(candidates, n, N), n, N)
end

