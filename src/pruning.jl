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
# Distance pruning: Prune the states that cannot reach the final state
# before the end of the sequence

struct DistancePruning <: PruningStrategy
    distances::Dict
end

function DistancePruning(fsm::AbstractFSM)
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
    DistancePruning(distances)
end

function (pruning::DistancePruning)(candidates, n, N)
    function filt(p)
        if isfinal(p.first) return true end
        dist = get(pruning.distances, p.first, 1)
        if dist - 1 ≤ N-n return true end
        return false
    end
    filter!(filt, candidates)
end

#######################################################################
# Threshold pruning: Prune the activate states that have weights lower
# than a given threshold

struct ThresholdPruning <: PruningStrategy
    Δ::Float64
end

function (pruning::ThresholdPruning)(candidates, n, N)
    maxval = maximum(p -> p.second, candidates)
    filter!(p -> maxval - p.second ≤ pruning.Δ, candidates)
end

