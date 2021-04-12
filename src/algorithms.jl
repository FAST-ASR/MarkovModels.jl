# MarkovModels.jl
#
# Lucas Ondel, 2021

function αrecursion!(α::AbstractMatrix{T}, π::AbstractVector{T},
                     A::AbstractMatrix{T}, lhs::AbstractMatrix;
                     prune!::Function = identity) where T
    N = size(lhs, 2)
    Aᵀ = transpose(A)
    α[:, 1] = lhs[:, 1] .* π

    for n in 2:N
        # Equivalent to:
        #αₙ = (Aᵀ * α[:,n-1]) .* lhs[:, n]
        αₙ = mul!(similar(α[:, n], T, size(Aᵀ, 1)), Aᵀ,
                  α[:,n-1] .* lhs[:,n], one(T), zero(T))

        #prune!(αₙ, n)
        α[:, n] = αₙ
    end
    α
end

function βrecursion!(β::AbstractMatrix{T}, ω::AbstractVector{T},
                     A::AbstractMatrix{T}, lhs::AbstractMatrix;
                     prune!::Function = identity) where T
    N = size(lhs, 2)
    β[:, end] = ω
    for n in N-1:-1:1
        # Equivalent to:
        #βₙ = A * (β[:, n+1] .* lhs[:, n+1])
        βₙ = mul!(similar(β[:, n], T, size(A, 2)), A,
                  β[:,n+1] .* lhs[:,n+1], one(T), zero(T))

        #prune!(βₙ, n)
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
    αβrecursion(pinit, pend, A, lhs[, pruning = nopruning])

Baum-Welch algorithm where `pinit` is initial state probabilities,
`pfinal` is the final states probabilities and `A` is the transition
matrix and `lh` is the per-state and per-frame likelihood.
"""
function αβrecursion(pinit::AbstractVector{T},
                     pfinal::AbstractVector{T},
                     A::AbstractMatrix{T},
                     lhs::AbstractMatrix{T};
                     pruning::T = upperbound(T),
                    ) where T
    S, N = length(π), size(lhs, 2)
    α = zeros(T, S, N)
    β = zeros(T, S, N)
    γ = zeros(T, S, N)

    αrecursion!(α, pinit, A, lhs,
                prune! = (αₙ, n) -> prune_α!(αₙ, cfsm.distances, N, n, pruning))
    βrecursion!(β, pfinal, A, lhs,
                prune! = (βₙ, n) -> prune_β!(βₙ, n, pruning, α))

    αβ = α .* β
    αβ ./ sum(αβ, dims = 1)
end
