# Implementation of generic graph algorithms.

using StatsFuns

#######################################################################
# Different strategy to prune the graph during inference.

export PruningStrategy
export ThresholdPruning


abstract type PruningStrategy end

# No pruning strategy for the αβ recursion
struct NoPruning <: PruningStrategy end
const nopruning = NoPruning()
(::NoPruning)(candidates::Dict{State, T}) where T <: AbstractFloat = candidates

# Pruning strategy which retain all the hypotheses highger than a
# given threshold
struct ThresholdPruning <: PruningStrategy
    Δ::Real
end

function (pruning::ThresholdPruning)(candidates::Dict{State, T}) where T <: AbstractFloat
    maxval = maximum(p -> p.second, candidates)
    return filter(p -> maxval - p.second ≤ pruning.Δ, candidates)
end



#######################################################################
# Baum-Welch (forward-backward) algorithm

export αrecursion
export βrecursion
export αβrecursion

"""
    αrecursion(graph, llh[, pruning = ...])

Forward step of the Baum-Welch algorithm in the log-domain.
"""
function αrecursion(g::AbstractGraph, llh::Matrix{T};
                    pruning::Union{Real, NoPruning} = nopruning) where T <: AbstractFloat
    pruning = pruning ≠ nopruning ? ThresholdPruning(pruning) : pruning
    α = Matrix{T}(undef, size(llh))
    fill!(α, T(-Inf))

    activestates = Dict{State, T}(initstate(g) => T(0.0))
    newstates = Dict{State, T}()
    for n in 1:size(llh, 2)
        for state_weightpath in activestates
            state, weightpath = state_weightpath
            for nstate_linkweight in emittingstates(forward, state)
                nstate, linkweight = nstate_linkweight
                nweightpath = weightpath + linkweight + llh[nstate.pdfindex, n]
                newstates[nstate] = logaddexp(get(newstates, nstate, T(-Inf)), nweightpath)
            end
        end

        for nstate_nweightpath in newstates
            nstate, nweightpath = nstate_nweightpath
            α[nstate.pdfindex, n] = logaddexp(α[nstate.pdfindex, n], nweightpath)
        end

        empty!(activestates)
        merge!(activestates, pruning(newstates))
        empty!(newstates)
    end
    α
end

"""
    βrecursion(graph, llh[, pruning = ...])

Backward step of the Baum-Welch algorithm in the log domain.
"""
function βrecursion(g::AbstractGraph, llh::Matrix{T};
                    pruning::Union{Real, NoPruning} = nopruning) where T <: AbstractFloat
    pruning = pruning ≠ nopruning ? ThresholdPruning(pruning) : pruning
    β = Matrix{eltype(llh)}(undef, size(llh))
    fill!(β, T(-Inf))

    activestates = Dict{State, T}(finalstate(g) => T(0.0))
    newstates = Dict{State, T}()

    for n in size(llh, 2):-1:1
        for state_weightpath in activestates
            state, weightpath = state_weightpath
            emitting = isemitting(state)
            prev_llh = emitting ? llh[state.pdfindex, n+1] : T(0.0)
            for nstate_linkweight in emittingstates(backward, state)
                nstate, linkweight = nstate_linkweight
                nweightpath = weightpath + linkweight + prev_llh
                newstates[nstate] = logaddexp(get(newstates, nstate, T(-Inf)), nweightpath)
            end
        end

        for nstate_nweightpath in newstates
            nstate, nweightpath = nstate_nweightpath
            β[nstate.pdfindex, n] = logaddexp(β[nstate.pdfindex, n], nweightpath)
        end

        empty!(activestates)
        merge!(activestates, pruning(newstates))
        empty!(newstates)
    end
    β
end

"""
    αβrecursion(graph, llh[, pruning = ...])

Baum-Welch algorithm in  the log domain.
"""
function αβrecursion(g::AbstractGraph, llh::Matrix{T};
                     pruning::Union{Real, NoPruning} = nopruning) where T <: AbstractFloat
    α = αrecursion(g, llh, pruning = pruning)
    β = βrecursion(g, llh, pruning = pruning)
    α + β .- logsumexp(α + β, dims = 1)
end

#######################################################################
# Determinization algorithm

export determinize

"""
    determinize(graph)


Create a new graph where each states are connected by at most one link.
"""
function determinize(g::Graph)
    newg = Graph()

    newstates = Dict{StateID, State}()
    for state in states(g)
        newstates[id(state)] = State(id(state), pdfindex(state))
    end

    newarcs = Dict{Tuple{State, State}, Real}()
    for state in states(g)
        for link in children(state)
            src = newstates[id(state)]
            dest = newstates[id(link.dest)]
            destweight = get(newarcs, (src, dest), oftype(link.weight, -Inf))
            newweight = logaddexp(destweight, link.weight)
            newarcs[(src, dest)] = newweight
        end
    end

    for state in values(newstates) addstate!(newg, state) end
    for arc in newarcs link!(arc[1][1], arc[1][2], arc[2]) end

    newg
end

#######################################################################
# Normalization algorithm

export weightnormalize

"""
    normalize(graph)

Update the weights of the graph such that the exponentiation of the
weight of all the outoing arc from a state sum up to one.
"""
function weightnormalize(g::Graph)
    newg = Graph()

    newstates = Dict{StateID, State}()
    for state in states(g)
        newstates[id(state)] = State(id(state), pdfindex(state))
    end

    newarcs = Vector{Tuple{State, State, Real}}()
    for state in states(g)
        lognorm = reduce(logaddexp, [link.weight for link in children(state)], init=-Inf)
        for link in children(state)
            src = newstates[id(state)]
            dest = newstates[id(link.dest)]
            push!(newarcs, (src, dest, link.weight - lognorm))
        end
    end

    for state in values(newstates) addstate!(newg, state) end
    for arc in newarcs link!(arc[1], arc[2], arc[3]) end

    newg
end

