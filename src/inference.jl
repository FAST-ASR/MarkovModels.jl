# Implementation of generic graph algorithms.

using StatsFuns: logaddexp, logsumexp
import Base: union

#######################################################################
# Different strategy to prune the graph during inference.

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

function (pruning::ThresholdPruning)(
    candidates::Dict{State, T}
) where T <: AbstractFloat
    maxval = maximum(p -> p.second, candidates)
    filter!(p -> maxval - p.second ≤ pruning.Δ, candidates)
end

#######################################################################
# Baum-Welch (forward-backward) algorithm

"""
    αrecursion(graph, llh[, pruning = ...])

Forward step of the Baum-Welch algorithm in the log-domain.
"""
function αrecursion(
    fsm::FSM,
    llh::Matrix{T};
    pruning::Union{Real, NoPruning} = nopruning
) where T <: AbstractFloat
    pruning! = pruning ≠ nopruning ? ThresholdPruning(pruning) : pruning

    activestates = Dict{State, T}(initstate(fsm) => T(0.0))
    α = Vector{Dict{State, T}}()
    for n in 1:size(llh, 2)
        push!(α, Dict{State,T}())
        for (state, weightpath) in activestates
            for (nstate, linkweight) in emittingstates(fsm, state, forward)
                nweightpath = weightpath + linkweight
                α[n][nstate] = logaddexp(get(α[n], nstate, T(-Inf)), nweightpath)
            end
        end

        # Add the log-likelihood outside the loop to add it only once
        for s in keys(α[n]) α[n][s] += llh[s.pdfindex, n] end

        empty!(activestates)
        merge!(activestates, pruning!(α[n]))
    end
    α
end

"""
    βrecursion(graph, llh[, pruning = ...])

Backward step of the Baum-Welch algorithm in the log domain.
"""
function βrecursion(
    fsm::FSM,
    llh::Matrix{T};
    pruning::Union{Real, NoPruning} = nopruning
) where T <: AbstractFloat

    pruning! = pruning ≠ nopruning ? ThresholdPruning(pruning) : pruning

    activestates = Dict{State, T}()
    β = Vector{Dict{State, T}}()
    push!(β, Dict(s => T(0.0)
         for (s, w) in emittingstates(fsm, finalstate(fsm), backward)))

    for n in size(llh, 2)-1:-1:1
        # Update the active tokens
        empty!(activestates)
        merge!(activestates, pruning!(β[1]))

        pushfirst!(β, Dict{State,T}())
        for (state, weightpath) in activestates
            prev_llh = llh[state.pdfindex, n+1]
            for (nstate, linkweight) in emittingstates(fsm, state, backward)
                nweightpath = weightpath + linkweight + prev_llh
                β[1][nstate] = logaddexp(get(β[1], nstate, T(-Inf)), nweightpath)
            end
        end
    end
    β
end

"""
    αβrecursion(graph, llh[, pruning = ...])

Baum-Welch algorithm per state in the log domain.
"""
function αβrecursion(
    fsm::FSM, llh::Matrix{T};
    pruning::Union{Real, NoPruning} = nopruning
) where T <: AbstractFloat

    lnα = αrecursion(fsm, llh, pruning = pruning)
    lnβ = βrecursion(fsm, llh, pruning = pruning)

    lnγ = Vector{Dict{State,T}}()

    ttl = 0.
    for n in 1:size(llh, 2)
        push!(lnγ, Dict{State, T}())
        for s in intersect(keys(lnα[n]), keys(lnβ[n]))
            lnγ[n][s] = lnα[n][s] + lnβ[n][s]
        end

        sum = logsumexp(values(lnγ[n]))
        for s in keys(lnγ[n]) lnγ[n][s] -= sum end

        # We could use any other time step to get the total
        # log-likelihood but, in case of heavy pruning, it is probably
        # safer to get it from the last frame.
        if n == size(llh, 2) ttl = sum end
    end

    lnγ, ttl
end

"""
    resps(fsm, lnαβ[, dense = false])

Convert the output of the `αβrecursion` to the per-frame pdf
responsibilities. When `dense = false` (default) the function returns
a `Dict{Pdfindex, Vector}. If `dense = true`, the function returns
a matrix of size SxN where S is the number of unique pdfs.
"""
function resps(
    fsm::FSM,
    lnαβ::Vector{Dict{State, T}};
    dense = false,
) where T <: AbstractFloat
    N = length(lnαβ)

    γ = Dict{Pdfindex, Vector}()
    for n in 1:N
        for (s, w) in lnαβ[n]
            γ_s = get(γ, s.pdfindex, zeros(N))
            γ_s[n] += exp(w)
            γ[s.pdfindex] = γ_s
        end
    end

    if ! dense
        return γ
    end

    # Build the dense matrix
    S = length(Set([s.pdfindex for s in emittingstates(fsm)]))
    denseγ = zeros(T, S, N)
    for idx in keys(γ)
        denseγ[idx, :] = γ[idx]
    end
    denseγ
end

function maxβrecursion(
    g::FSM,
    llh::Matrix{T},
    α::Vector{Dict{State,T}}
) where T <: AbstractFloat

    bestseq = Vector{State}()
    activestates = Dict{State, T}(finalstate(g) => T(0.0))
    newstates = Dict{State, T}()

    for n in size(llh, 2):-1:1
        for (state, weightpath) in activestates
            emitting = isemitting(state)
            prev_llh = emitting ? llh[state.pdfindex, n+1] : T(0.0)
            for (nstate, linkweight) in emittingstates(g, state, backward)
                nweightpath = weightpath + linkweight + prev_llh
                newstates[nstate] = logaddexp(get(newstates, nstate, T(-Inf)), nweightpath)
            end
        end

        hypscores = Vector{T}(undef, length(newstates))
        hypstates = Vector{State}(undef, length(newstates))
        for (i, (nstate, nweightpath)) in enumerate(newstates)
            hypscores[i] = get(α[n], nstate, T(-Inf)) + nweightpath
            hypstates[i] = nstate
        end
        println(hypstates)
        println(hypscores)
        println("-----")
        maxval, maxidx = findmax(hypscores)
        best = hypstates[maxidx]
        pushfirst!(bestseq, best)

        empty!(activestates)
        activestates[best] = newstates[best]
        empty!(newstates)
    end
    bestseq
end

"""
    viterbi(graph, llh[, pruning = ...])

Viterbi algorithm.
"""
function viterbi(
    fsm::FSM,
    llh::Matrix{T};
    pruning::Union{Real, NoPruning} = nopruning
) where T <: AbstractFloat

    α = αrecursion(fsm, llh, pruning = pruning)
    path = maxβrecursion(fsm, llh, α)

    # Return the best seq as a new graph
    ng = FSM()
    prevstate = initstate(ng)
    for state in path
        s = addstate!(ng, pdfindex = state.pdfindex, label = state.label)
        link!(ng, prevstate, s)
        prevstate = s
    end
    link!(ng, prevstate, finalstate(ng), 0.)
    ng
end

