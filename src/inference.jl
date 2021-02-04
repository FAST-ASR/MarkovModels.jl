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
            for (nstate, linkweight) in nextemittingstates(state)
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
                    pruning::PruningStrategy = nopruning) where T <: AbstractFloat
    N = size(llh, 2)

    fsmᵀ = fsm |> transpose

    activestates = Dict{State, T}()
    β = Vector{Dict{State, T}}()
    push!(β, Dict(s => T(0.0) for (s, w) in nextemittingstates(initstate(fsmᵀ))))
    for n in N-1:-1:1

        # Update the active tokens
        empty!(activestates)
        merge!(activestates, pruning(β[1], n+1, N))

        pushfirst!(β, Dict{State,T}())
        for (state, weightpath) in activestates
            prev_llh = llh[state.pdfindex, n+1]
            for (nstate, linkweight) in nextemittingstates(state)
                nweightpath = weightpath + linkweight + prev_llh
                β[1][nstate] = logaddexp(get(β[1], nstate, T(-Inf)), nweightpath)
            end
        end
    end
    β
end

"""
    αβrecursion(fsm, llh[, pruning = nopruning])

Baum-Welch algorithm per state in the log domain. This function returns
a tuple `(lnαβ, ttl)` where `lnαβ` is a sparse representation of the
responsibilities and `ttl` is the log-likelihood of the sequence given
the inference `fsm`.
"""
function αβrecursion(fsm::AbstractFSM, llh::AbstractMatrix{T};
                     pruning::PruningStrategy = nopruning) where T <: AbstractFloat

    lnα = αrecursion(fsm, llh, pruning = pruning)
    lnβ = βrecursion(fsm, llh, pruning = BackwardPruning(lnα))

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

struct Responsibilities
    N::Int64
    sparseresps::Dict{PdfIndex, Vector{<:AbstractFloat}}
end

Base.getindex(r::Responsibilities, i) = get(r.sparseresps, i, zeros(r.N))

"""
    resps(fsm, lnαβ[, dense = false])

Convert the output of the `αβrecursion` to the per-frame pdf
responsibilities. The returned value is a dictionary whose keys are
pdf indices.
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
    Responsibilities(N, γ)
end

"""
    maxβrecursion(fsm, lnα)

Compute bestpath using forward pass.
"""
function maxβrecursion(fsm::AbstractFSM{T}, lnα, draw, labelfilter) where T <: AbstractFloat
    fsmᵀ = fsm |> transpose
    prevstate = initstate(fsmᵀ)
    labels = []
    for n in length(lnα):-1:1
        m = T(-Inf)

        weights = T[]
        candidates = []
        for (state, weight, path) in nextemittingstates(prevstate, return_path = true)
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

        # Extract the label sequence
        for state in bestpath
            (islabeled(state) && labelfilter(state.label)) || continue
            push!(labels, state.label)
        end

        if (islabeled(beststate) && labelfilter(beststate.label)) || continue
            push!(labels, beststate.label)
        end

    end

    reverse(labels)
end

"""
    beststring(fsm, llh[, pruning = nopruning, labelfilter = x -> true])

Returns the best sequence of the labels given the log-likelihood `llh`.
"""
function beststring(fsm::AbstractFSM, llh; pruning::PruningStrategy = nopruning,
                  labelfilter = x -> true)
    lnα = αrecursion(fsm, llh, pruning = pruning)
    maxβrecursion(fsm, lnα, false, labelfilter)
end

"""
    samplestring(fsm, llh[, nsamples = 1, pruning = nopruning, labelfilter = x -> true])

Sample a sequence of labels given the log-likelihood `llh`.

"""
function samplestring(fsm::AbstractFSM, llh; pruning::PruningStrategy = nopruning,
                    nsamples = 1, labelfilter = x -> true)
    lnα = αrecursion(fsm, llh, pruning = pruning)
    [maxβrecursion(fsm, lnα, true, labelfilter) for i in 1:nsamples]
end

