# SPDX-License-Identifier: MIT

"""
    αβrecursion(cfsm, lhs::AbstractMatrix{T}) where T

Return the states log conditional probabilities (as a matrix) and the
total likelihood of the sequence.

See also: [`compile`](@ref)
"""
function αβrecursion(cfsm, lhs::AbstractMatrix{T}) where T
    αβrecursion(cfsm.π, cfsm.ω, cfsm.A, cfsm.Aᵀ, lhs)
end

#######################################################################
# Generic implementation

function αrecursion!(α::AbstractMatrix{T}, π::AbstractVector{T},
                     Aᵀ::AbstractMatrix{T}, lhs::AbstractMatrix) where T
    N = size(lhs, 2)
    buffer = similar(lhs[:, 1])
    α[:, 1] = lhs[:, 1] .* π
    for n in 2:N
        # Equivalent to:
        #αₙ = (Aᵀ * α[:,n-1]) .* lhs[:, n]

        αₙ = view(α, :, n)
        broadcast!(*, buffer, view(α, :, n-1), view(lhs, :, n))
        mul!(αₙ, Aᵀ, buffer, one(T), zero(T))
    end
    α
end

function βrecursion!(β::AbstractMatrix{T}, ω::AbstractVector{T},
                     A::AbstractMatrix{T}, lhs::AbstractMatrix{T}) where T
    N = size(lhs, 2)
    buffer = similar(lhs[:, 1])
    β[:, end] = ω
    for n in N-1:-1:1
        # Equivalent to:
        #βₙ = A * (β[:, n+1] .* lhs[:, n+1])

        βₙ = view(β, :, n)
        broadcast!(*, buffer, view(β, :, n+1), view(lhs, :, n+1))
        mul!(βₙ, A, buffer, one(T), zero(T))
    end
    β
end

function αβrecursion(π::AbstractVector{T}, ω::AbstractVector{T},
                     A::AbstractMatrix{T}, Aᵀ::AbstractMatrix,
                     lhs::AbstractMatrix{T}) where T
    S, N = length(π), size(lhs, 2)
    α = fill!(similar(lhs, T, S, N), zero(T))
    β = fill!(similar(lhs, T, S, N), zero(T))
    γ = fill!(similar(lhs, T, S, N), zero(T))

    αrecursion!(α, π, Aᵀ, lhs)
    βrecursion!(β, ω, A, lhs)

    broadcast!(*, γ, α, β)
    sums = sum(γ, dims = 1)
    broadcast!(/, γ, γ, sums)

    γ, minimum(sums)
end

#######################################################################
# Sparse GPU matrix implementation

function αrecursion!(α::CuMatrix{T}, π::CuSparseVector{T},
                     Aᵀ::CuSparseMatrixCSR{T}, lhs::CuMatrix{T}) where T
    N = size(lhs, 2)
    buffer = similar(α[:,1], T, length(π))
    elmul_svdv!(view(α, :, 1), π, view(lhs, :, 1))
    for n in 2:N
        # Equivalent to:
        #αₙ = (Aᵀ * α[:,n-1]) .* lhs[:, n]

        mul_smdv!(view(α, :, n), Aᵀ, view(α, :, n-1) .* view(lhs, :, n))
    end
    α
end

function βrecursion!(β::CuMatrix{T}, ω::CuSparseVector{T},
                     A::CuSparseMatrixCSR{T}, lhs::CuArray{T}) where T
    N = size(lhs, 2)
    β[:, end] = ω
    for n in N-1:-1:1
        # Equivalent to:
        #βₙ = A * (β[:, n+1] .* lhs[:, n+1])
        #βₙ = mul!(similar(β[:, n], T, size(A, 2)), A, β[:,n+1] .* lhs[:, n+1])

        mul_smdv!(view(β, :, n), A, view(β, :, n+1) .* view(lhs, :, n+1))
    end
    β
end

