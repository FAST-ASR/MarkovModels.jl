# SPDX-License-Identifier: MIT

function αβrecursion(cfsm, lhs::AbstractMatrix{T}; pruning = T(Inf)) where T
    αβrecursion(cfsm.π, cfsm.ω, cfsm.A, cfsm.Aᵀ, lhs, pruning, cfsm.dists)
end

#######################################################################
# Pruning

function remove_invalid_αpath!(αₙ::AbstractVector{T}, distances, N, n) where T
    for i in 1:length(αₙ)
        if distances[i] > (N - n) αₙ[i] = zero(T) end
    end
    αₙ
end

function remove_invalid_αpath!(αₙ::SparseVector{T}, distances, N, n) where T
    I, V = findnz(αₙ)
    for i in I
        if distances[i] > (N - n) αₙ[i] = zero(T) end
    end
    αₙ
end

function prune_α!(αₙ::AbstractVector{T}, distances, N, n, threshold) where T
    remove_invalid_αpath!(αₙ, distances, N, n)
    maxval = maximum(αₙ)
    for i in 1:length(αₙ)
        αₙ[i] = (maxval/αₙ[i] ≤ threshold) ? αₙ[i] : zero(T)
    end
    αₙ
end

function prune_α!(αₙ::SparseVector{T}, distances, N, n, threshold) where T
    remove_invalid_αpath!(αₙ, distances, N, n)
    SparseArrays.fkeep!(αₙ, (i,v) -> maximum(αₙ)/v ≤ threshold)
end

function prune_β!(βₙ::AbstractVector{T}, n, threshold, α) where T
    I = findall(α[:, n] .> zero(T))
    maxval = maximum(βₙ)
    for i in 1:length(βₙ)
        βₙ[i] = i ∈ I && maxval/βₙ[i] ≤ threshold ? βₙ[i] : zero(T)
    end
    βₙ
end

function prune_β!(βₙ::SparseVector{T}, n, threshold, α) where T
    I, V = findnz(α[:, n])
    SparseArrays.fkeep!(βₙ, (i,v) -> i ∈ I && maximum(βₙ)/v ≤ threshold)
end

#######################################################################
# Dense matrix implementation

function αrecursion!(α::AbstractMatrix{T}, π::AbstractVector{T},
                     Aᵀ::AbstractMatrix{T}, lhs::AbstractMatrix,
                     prune!::Function) where T
    N = size(lhs, 2)
    buffer = similar(α[:,1], T, length(π))
    α[:, 1] = lhs[:, 1] .* π
    for n in 2:N
        # Equivalent to:
        #αₙ = (Aᵀ * α[:,n-1]) .* lhs[:, n]
        αₙ = mul!(buffer, Aᵀ, view(α, :, n-1) .* view(lhs, :, n), one(T), zero(T))
        prune!(αₙ, n)
        α[:, n] = αₙ
    end
    α
end

function βrecursion!(β::AbstractMatrix{T}, ω::AbstractVector{T},
                     A::AbstractMatrix{T}, lhs::AbstractMatrix{T},
                     prune!::Function) where T
    N = size(lhs, 2)
    buffer = similar(β[:,1], T, length(ω))
    β[:, end] = ω
    for n in N-1:-1:1
        # Equivalent to:
        #βₙ = A * (β[:, n+1] .* lhs[:, n+1])
        βₙ = mul!(buffer, A, view(β, :,n+1) .* view(lhs, :, n+1), one(T), zero(T))
        prune!(βₙ, n)
        β[:, n] = βₙ
    end
    β
end

function αβrecursion(π::AbstractVector{T}, ω::AbstractVector{T},
                     A::AbstractMatrix{T}, Aᵀ::AbstractMatrix,
                     lhs::AbstractMatrix{T},
                     pruning, distances) where T
    S, N = length(π), size(lhs, 2)
    α = similar(π, T, S, N)
    β = similar(π, T, S, N)
    γ = similar(π, T, S, N)

    pα = T(pruning) == T(Inf) ? (αₙ, n) -> αₙ : (
        (αₙ, n) -> prune_α!(αₙ, distances, N, n, T(pruning)))
    pβ = T(pruning) == T(Inf) ? (βₙ, n) -> βₙ : (
        (βₙ, n) -> prune_β!(βₙ, n, T(pruning), α))

    αrecursion!(α, π, Aᵀ, lhs, pα)
    βrecursion!(β, ω, A, lhs, pβ)

    αβ = α .* β
    sums = sum(αβ, dims = 1)
    αβ ./ sums, maximum(sums)
end

#######################################################################
# Sparse matrix implementation

function αrecursion!(α::AbstractSparseMatrix{T}, π::AbstractSparseVector{T},
                     Aᵀ::AbstractSparseMatrix{T}, lhs::AbstractMatrix{T},
                     prune!::Function) where T
    N = size(lhs, 2)
    α[:, 1] = lhs[:,1] .* π
    for n in 2:N
        # Equivalent to:
        #αₙ = (Aᵀ * α[:,n-1]) .* lhs[:, n]
        αₙ = mul!(similar(α[:,n], T, size(Aᵀ, 1)), Aᵀ,
                  α[:,n-1] .* lhs[:,n], one(T), zero(T))
        prune!(αₙ, n)
        α[:, n] = αₙ
    end
    α
end

function βrecursion!(β::AbstractSparseMatrix{T}, ω::AbstractSparseVector{T},
                     A::AbstractSparseMatrix{T}, lhs::AbstractMatrix{T},
                     prune!::Function) where T
    N = size(lhs, 2)
    β[:, end] = ω
    for n in N-1:-1:1
        # Equivalent to:
        #βₙ = A * (β[:, n+1] .* lhs[:, n+1])
        βₙ = mul!(similar(β[:, n], T, size(A, 2)), A,
                  β[:,n+1] .* lhs[:, n+1], one(T), zero(T))
        prune!(βₙ, n)
        β[:, n] = βₙ
    end
    β
end

function normalize_spmatrix(M::AbstractSparseMatrix{T}) where T
    sums = sum(M, dims = 1)
    I, J, V = findnz(M)
    for i in 1:length(V)
        V[i] /= sums[J[i]]
    end
    sparse(I, J, V), maximum(sums)
end

function αβrecursion(π::AbstractVector{T}, ω::AbstractVector{T},
                     A::AbstractSparseMatrix{T}, Aᵀ::AbstractSparseMatrix{T},
                     lhs::AbstractMatrix{T}, pruning, distances) where T
    S, N = length(π), size(lhs, 2)
    α = spzeros(T, S, N)
    β = spzeros(T, S, N)
    γ = spzeros(T, S, N)

    pα = T(pruning) == T(Inf) ? (αₙ, n) -> αₙ : (
        (αₙ, n) -> prune_α!(αₙ, distances, N, n, T(pruning)))
    pβ = (βₙ, n) -> prune_β!(βₙ, n, T(pruning), α)

    αrecursion!(α, π, Aᵀ, lhs, pα)
    βrecursion!(β, ω, A, lhs, pβ)

    normalize_spmatrix(α .* β)
end

#######################################################################
# Sparse GPU matrix implementation

function αrecursion!(α::CuMatrix{T}, π::CuSparseVector{T},
                     Aᵀ::CuSparseMatrixCSR{T}, lhs::CuMatrix{T},
                     prune!::Function) where T
    N = size(lhs, 2)
    buffer = similar(α[:,1], T, length(π))
    #α[:, 1] = lhs[:, 1] .* π
    el_mul!(view(α, :, 1), π, view(lhs, :, 1))
    for n in 2:N
        # Equivalent to:
        #αₙ = (Aᵀ * α[:,n-1]) .* lhs[:, n]
        #αₙ = mul!(buffer, Aᵀ, view(α, :, n-1) .* view(lhs, :, n))
        α[:, n] = smdv!(buffer, Aᵀ, view(α, :, n-1) .* view(lhs, :, n))
    end
    α
end

function βrecursion!(β::CuMatrix{T}, ω::CuSparseVector{T},
                     A::CuSparseMatrixCSR{T}, lhs::CuArray{T},
                     prune!::Function) where T
    N = size(lhs, 2)
    β[:, end] = ω
    for n in N-1:-1:1
        # Equivalent to:
        #βₙ = A * (β[:, n+1] .* lhs[:, n+1])
        #βₙ = mul!(similar(β[:, n], T, size(A, 2)), A, β[:,n+1] .* lhs[:, n+1])
        smdv!(β[:, n], A, view(β, :, n+1) .* view(lhs, :, n+1))
    end
    β
end

function normalize_spmatrix(M::CuSparseMatrix{T}) where T
    sums = sum(M, dims = 1)
    I, J, V = findnz(M)
    for i in 1:length(V)
        V[i] /= sums[J[i]]
    end
    sparse(I, J, V), maximum(sums)
end

#function αβrecursion(π::CuSparseVector{T}, ω::CuSparseVector{T},
#                     A::CuSparseMatrixCSC{T}, Aᵀ::CuSparseMatrixCSC{T},
#                     lhs::CuArray{T}, pruning, distances) where T
#    S, N = length(π), size(lhs, 2)
#    α = [CuSparseVector(spzeros(T, S)) for n in 1:N]
#    β = [CuSparseVector(spzeros(T, S)) for n in 1:N]
#    γ = CuSparseMatrixCSC(spzeros(T, S, N))
#
#    pα = T(pruning) == T(Inf) ? (αₙ, n) -> αₙ : (
#        (αₙ, n) -> prune_α!(αₙ, distances, N, n, T(pruning)))
#    pβ = (βₙ, n) -> prune_β!(βₙ, n, T(pruning), α)
#
#    αrecursion!(α, π, Aᵀ, lhs, pα)
#    βrecursion!(β, ω, A, lhs, pβ)
#
#    normalize_spmatrix(α .* β)
#end

function αβrecursion(π::CuSparseVector{T}, ω::CuSparseVector{T},
                     A::CuSparseMatrixCSR{T}, Aᵀ::CuSparseMatrixCSR{T},
                     lhs::CuArray{T}, pruning, distances) where T
    S, N = length(π), size(lhs, 2)
    α = similar(lhs, T, S, N)
    β = similar(lhs, T, S, N)
    γ = similar(lhs, T, S, N)

    pα = T(pruning) == T(Inf) ? (αₙ, n) -> αₙ : (
        (αₙ, n) -> prune_α!(αₙ, distances, N, n, T(pruning)))
    pβ = T(pruning) == T(Inf) ? (βₙ, n) -> βₙ : (
        (βₙ, n) -> prune_β!(βₙ, n, T(pruning), α))

    αrecursion!(α, π, Aᵀ, lhs, pα)
    βrecursion!(β, ω, A, lhs, pβ)

    αβ = α .* β
    sums = sum(αβ, dims = 1)
    αβ ./ sums, maximum(sums)
end

