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

"""
    ωrecursion(fsm, llh [, pruning = nopruning]}

Max-sum algorithm for finding the most probable path in the `fsm`
based on the log-likelihoods `llh`.

Returns the maximum weigths for each state per frame and the corresponding path.
"""
function ωrecursion(g::FSM, llh::Matrix{T}; pruning::Union{Real, NoPruning} = nopruning) where T <: AbstractFloat
    pruning! = pruning ≠ nopruning ? ThresholdPruning(pruning) : pruning
    
    activestates = Dict{State, T}(initstate(g) => T(0.0))
    # Weights per state and per frame 
    ω = Vector{Dict{State, T}}() 
    # Partial path how we reach the state in each frame
    ψ = Vector{Dict{State, Tuple{State, Vector}}}() 
    
    for n in 1:size(llh, 2)
        push!(ω, Dict{State, T}())
        push!(ψ, Dict{State, Tuple{State, Vector}}())
        for (state, weightpath) in activestates
            for(nstate, linkweight, path) in emittingstates(g, state, forward)
                nweightpath = weightpath + linkweight
                m = max(get(ω[n], nstate, T(-Inf)), nweightpath)
                ω[n][nstate] = m
                if m === nweightpath 
                    # Update the best path for nstate
                    ψ[n][nstate] = (state, path) # path: state -> nstate
                end
            end
        end
        for s in keys(ω[n]) ω[n][s] += llh[s.pdfindex, n] end
        
        empty!(activestates)
        merge!(activestates, pruning!(ω[n]))
    end
    
    # Remove emiting states with no direct connection with the final state
    fes = finalemittingstates(g)
    filter!(ψ[end]) do p
        haskey(fes, p.first)
    end
    filter!(ω[end]) do p
        haskey(fes, p.first)
    end
    
    # Add path from last emiting state to FSM's final state
    for s in keys(ψ[end])
        (_, path) = ψ[end][s]
        append!(path, fes[s])
    end
    
    ω,ψ
end

"""
    viterbi(fsm, llh [, pruning = nopruning])

Viterbi algorithm for finding the best path.
"""
function viterbi(g::FSM, llh::Matrix{T}; pruning::Union{Real, NoPruning} = nopruning) where T <: AbstractFloat

    @warn "viterbi is depracated! Use bestpath instead."
    lnω, ψ = ωrecursion(g, llh; pruning = pruning)
    
    bestpath = Vector{Link}()
    _, state = findmax(lnω[end])
    for n in length(ψ):-1:1
        state, path = ψ[n][state]
        prepend!(bestpath, path)
    end
    
    ng = FSM()
    prevs = initstate(ng)
    for l in bestpath
        s = addstate!(ng, pdfindex = l.dest.pdfindex, label = l.dest.label)
        #link!(ng, prevs, s, l.weight)
        link!(ng, prevs, s)
        prevs = s
    end
    link!(ng, prevs, finalstate(ng))
    
    ng |> removenilstates!
end

"""
    maxβrecursion(fsm, lnα)

Compute bestpath using forward pass.
"""
function maxβrecursion(
    g::FSM,
    lnα::Vector{Dict{State, T}}) where T <: AbstractFloat
    
    π = FSM()
    prevs = finalstate(g)
    lasts = finalstate(π)

    for n in size(lnα, 1):-1:1
        m = T(-Inf)
        bests = nothing
        bestp = nothing
        for (state, weight, path) in emittingstates(g, prevs, backward)
            hyp = weight + get(lnα[n], state, T(-Inf))
            if (hyp > m)
                m = hyp
                bests = state
                bestp = path
            end
        end
        prevs = bests

        for l in bestp
            s = addstate!(π, pdfindex = l.dest.pdfindex, label = l.dest.label)
            link!(π, s, lasts)
            lasts = s
        end
    end

    link!(π, initstate(π), lasts)

    π
end

"""
    bestpath(fsm, llh [, pruning = nopruning])

Lucas's algorithm for finding the best path.

It uses forward pass from forward-backward.
"""
function bestpath(g::FSM, llh::Matrix{T}; pruning::Union{Real, NoPruning} = nopruning) where T <: AbstractFloat
    lnα = αrecursion(g, llh, pruning = pruning)
    maxβrecursion(g, lnα)
end


