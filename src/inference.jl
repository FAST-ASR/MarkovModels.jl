# MarkovModels.jl
#
# Lucas Ondel, 2021

function αrecursion!(α::AbstractMatrix{T}, A::AbstractMatrix{T},
                     π::AbstractVector{T}, state_lhs::AbstractMatrix;
                     prune!::Function = identity) where T<:SemiField
    N = size(state_lhs, 2)
    Aᵀ = transpose(A)
    α[:, 1] = state_lhs[:, 1] .* π

    for n in 2:N
        # Equivalent to:
        #αₙ = (Aᵀ * α[:,n-1]) .* state_lhs[:, n]
        αₙ = mul!(similar(α[:, n], T, size(Aᵀ, 1)), Aᵀ,
                  α[:,n-1] .* state_lhs[:,n], one(T), zero(T))

        prune!(αₙ, n)
        α[:, n] = αₙ
    end
    α
end

function βrecursion!(β::AbstractMatrix{T}, A::AbstractMatrix{T},
                     ω::AbstractVector{T}, state_lhs::AbstractMatrix;
                     prune!::Function = identity) where T<:SemiField
    N = size(state_lhs, 2)
    β[:, end] = ω
    for n in N-1:-1:1
        # Equivalent to:
        #βₙ = A * (β[:, n+1] .* state_lhs[:, n+1])
        βₙ = mul!(similar(β[:, n], T, size(A, 2)), A,
                  β[:,n+1] .* state_lhs[:,n+1], one(T), zero(T))

        prune!(βₙ, n)
        β[:, n] = βₙ
    end
    β
end

function remove_invalid_αpath!(αₙ::AbstractVector{T}, distances, N, n) where T
    for (i,v) in zip(findnz(αₙ)...)
        if distances[i] > (N - n + 1)
            αₙ[i] = zero(T)
        end
    end
    αₙ
end

function prune_α!(αₙ, distances, N, n, threshold)
    #remove_invalid_αpath!(αₙ, distances, N, n)
    SparseArrays.fkeep!(αₙ, (i,v) -> maximum(αₙ)/v ≤ threshold)
end

function prune_β!(βₙ, n, threshold, α)
    I, V = findnz(α[:, n])
    SparseArrays.fkeep!(βₙ, (i,v) -> i ∈ I && maximum(βₙ)/v ≤ threshold)
end


"""
    αβrecursion(cfsm, lhs[, pruning = nopruning])

Baum-Welch algorithm. This function returns
a tuple `(lnαβ, ttl)` where `lnαβ` is a sparse representation of the
responsibilities and `ttl` is the log-likelihood of the sequence given
the inference `fsm`.
"""
function αβrecursion(cfsm::CompiledFSM{T},
                     lhs::AbstractMatrix{T};
                     pruning::T = upperbound(T),
                    ) where T<:SemiField
    # Expand the per-pdf log-likelihoods to the per-state
    # likelihoods. This is necessary when some states share the
    # same emission.
    state_lhs = cfsm.state_pdf * lhs

    S, N = nstates(cfsm), size(lhs, 2)
    α = spzeros(T, S, N)
    β = spzeros(T, S, N)
    γ = spzeros(T, S, N)

    αrecursion!(α, cfsm.A, cfsm.π, state_lhs,
                prune! = (αₙ, n) -> prune_α!(αₙ, cfsm.distances, N, n, pruning))
    βrecursion!(β, cfsm.A, cfsm.ω, state_lhs,
                prune! = (βₙ, n) -> prune_β!(βₙ, n, pruning, α))

    αβ = α .* β
    αβ ./ sum(αβ, dims = 1)
end

"""
    resps(cfsm, lnγ)

Convert the output of the `αβrecursion` to the per-frame pdf
responsibilities. By default, the results
"""

function resps(cfsm::CompiledFSM, lnγ::AbstractMatrix{<:SemiField})
    cfsm.state_pdf' * convert(Float64, lnγ)
end

"""
    maxβrecursion(fsm, lnα)

Compute bestpath using forward pass.
"""
function maxβrecursion(fsm::FSM{T}, lnα, draw, labelfilter) where T<:SemiField
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
    beststring(fsm, lhs[, pruning = nopruning, labelfilter = x -> true])

Returns the best sequence of the labels given the log-likelihood `lhs`.
"""
#function beststring(fsm::AbstractFSM, lhs; pruning::PruningStrategy = nopruning,
#                  labelfilter = x -> true)
#    lnα = αrecursion(fsm, lhs, pruning = pruning)
#    maxβrecursion(fsm, lnα, false, labelfilter)
#end

"""
    samplestring(fsm, lhs[, nsamples = 1, pruning = nopruning, labelfilter = x -> true])

Sample a sequence of labels given the log-likelihood `lhs`.
"""
#function samplestring(fsm::AbstractFSM, lhs; pruning::PruningStrategy = nopruning,
#                    nsamples = 1, labelfilter = x -> true)
#    lnα = αrecursion(fsm, lhs, pruning = pruning)
#    [maxβrecursion(fsm, lnα, true, labelfilter) for i in 1:nsamples]
#end

