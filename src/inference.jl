# Implementation of generic graph algorithms.

"""
    αrecursion(graph, llh[, pruning = ...])

Forward step of the Baum-Welch algorithm in the log-domain.
"""
function αrecursion(fsm::AbstractFSM, llh::AbstractMatrix{T};
                    pruning::PruningStrategy = nopruning) where T <: AbstractFloat

    N = size(llh, 2) # Number of observations
    α = Vector{Dict{State,T}}()
    activestates = Dict{State, T}(initstate(fsm) => T(0.0))
    for n in 1:N
        push!(α, Dict{State,T}())
        for (state, weightpath) in activestates
            for (nstate, linkweight, _) in nextemittingstates(state)
                nweightpath = weightpath + linkweight
                α[n][nstate] = logaddexp(get(α[n], nstate, T(-Inf)), nweightpath)
            end
        end

        # Add the log-likelihood outside the loop to add it only once
        for s in keys(α[n])
            α[n][s] += llh[s.pdfindex, n]
        end

        empty!(activestates)
        merge!(activestates, pruning(α[n], n, N))
    end
    α
end

"""
    βrecursion(graph, llh[, pruning = ...])

Backward step of the Baum-Welch algorithm in the log domain.
"""
function βrecursion(fsm::AbstractFSM, llh::AbstractMatrix{T};
                    pruning!::PruningStrategy = nopruning) where T <: AbstractFloat

    N = size(llh, 2)

    fsmᵀ = fsm |> transpose

    activestates = Dict{State, T}()
    β = Vector{Dict{State, T}}()
    push!(β, Dict(s => T(0.0) for (s, w) in nextemittingstates(initstate(fsmᵀ))))
    for n in N-1:-1:1
        # Update the active tokens
        empty!(activestates)
        merge!(activestates, pruning!(β[1], n, N))

        pushfirst!(β, Dict{State,T}())
        for (state, weightpath) in activestates
            prev_llh = llh[state.pdfindex, n+1]
            for (nstate, linkweight, _) in nextemittingstates(state)
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
function αβrecursion(fsm::AbstractFSM, llh::AbstractMatrix{T};
                     pruning!::PruningStrategy = nopruning) where T <: AbstractFloat

    lnα = αrecursion(fsm, llh, pruning = pruning!)
    lnβ = βrecursion(fsm, llh, pruning! = pruning!)

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
function resps(fsm::AbstractFSM, lnαβ)
    N = length(lnαβ)

    γ = Dict{PdfIndex, Vector}()
    for n in 1:N
        for (s, w) in lnαβ[n]
            γ_s = get(γ, s.pdfindex, zeros(N))
            γ_s[n] += exp(w)
            γ[s.pdfindex] = γ_s
        end
    end
    γ
end

"""
    maxβrecursion(fsm, lnα)

Compute bestpath using forward pass.
"""
function maxβrecursion(fsm::AbstractFSM{T}, lnα, draw, statefilter) where T <: AbstractFloat
    π = FSM{T}()

    fsmᵀ = fsm |> transpose

    prevstate = initstate(fsmᵀ)
    laststate = finalstate(π)

    for n in length(lnα):-1:1
        m = T(-Inf)

        weights = T[]
        candidates = []
        norm =
        for (state, weight, path) in nextemittingstates(prevstate)
            hyp = weight + get(lnα[n], state, -Inf)
            push!(weights, weight + get(lnα[n], state, -Inf))
            push!(candidates, (state, path))
        end
        norm = logsumexp(weights)
        weights = exp.(weights .- norm)

        if ! draw
            _, i = findmax(weights)
            beststate, bestpath = candidates[i]
        else
            beststate, bestpath = sample(candidates, Weights(weights))
        end
        prevstate = beststate

        for state in bestpath
            statefilter(state) || continue
            nstate = addstate!(π, pdfindex = state.pdfindex, label = state.label)
            link!(nstate, laststate)
            laststate = nstate
        end

        if statefilter(beststate)
            nstate = addstate!(π, pdfindex = beststate.pdfindex, label = beststate.label)
            link!(nstate, laststate)
            laststate = nstate
        end

    end
    link!(initstate(π), laststate)

    π
end

"""
    bestpath(fsm, llh [, pruning = nopruning])

Lucas's algorithm for finding the best path.

It uses forward pass from forward-backward.
"""
function bestpath(fsm::AbstractFSM, llh; pruning::PruningStrategy = nopruning,
                  statefilter = x -> true)
    lnα = αrecursion(fsm, llh, pruning = pruning)
    maxβrecursion(fsm, lnα, false, statefilter)
end

"""
    samplepath(fsm, llh [, pruning = nopruning])

Sample a state path.
"""
function samplepath(fsm::AbstractFSM, llh; pruning::PruningStrategy = nopruning,
                    size = 1, statefilter = x -> true)
    lnα = αrecursion(fsm, llh, pruning = pruning)
    [maxβrecursion(fsm, lnα, true, statefilter) for i in 1:size]
end


