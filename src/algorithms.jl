# SPDX-License-Identifier: MIT

const Abstract3DTensor{T} = AbstractArray{T,3} where T

#======================================================================
Forward recursion
======================================================================#

function αrecursion!(α::AbstractMatrix{T}, π::AbstractVector{T},
                     Aᵀ::AbstractMatrix{T}, lhs::AbstractMatrix) where T
    N = size(lhs, 2)
    buffer = similar(lhs[:, 1])
    @views elmul!(α[:,1], π, lhs[:,1])
    @views for n in 2:N
        matmul!(buffer, Aᵀ, α[:,n-1])
        elmul!(α[:,n], buffer, lhs[:,n])
    end
    α
end

function αrecursion!(α::Abstract3DTensor{T}, π::AbstractVector{T},
                     Aᵀ::AbstractMatrix{T}, lhs::Abstract3DTensor{T}) where T
    N = size(lhs, 2)
    buffer = similar(α[:,1,:])
    @views elmul!(α[:,1,:], π, lhs[:,1,:])
    @views for n in 2:N
        matmul!(buffer, Aᵀ, α[:,n,:])
        elmul!(α[:,n,:], buffer, lhs[:,n,:])
    end
    α
end

#======================================================================
Backward recursion
======================================================================#

function βrecursion!(β::AbstractMatrix{T}, ω::AbstractVector{T},
                     A::AbstractMatrix{T}, lhs::AbstractMatrix{T}) where T
    N = size(lhs, 2)
    buffer = similar(lhs[:, 1])
    β[:, end] = ω
    @views for n in N-1:-1:1
        elmul!(buffer, β[:,n+1], lhs[:,n+1])
        matmul!(β[:,n], A, buffer)
    end
    β
end

function βrecursion!(β::Abstract3DTensor{T}, ω::AbstractVector{T},
                     A::AbstractMatrix{T}, lhs::Abstract3DTensor{T}) where T
    N = size(lhs, 2)
    buffer = fill!(similar(β[:,1,:]), one(T))
    @views elmul!(β[:,end,:], ω, buffer)
    @views for n in N-1:-1:1
        elmul!(buffer, β[:,n+1,:], lhs[:,n+1,:])
        matmul!(β[:,n,:], A, buffer)
    end
    β
end

#======================================================================
Forward-Backward algorithm
======================================================================#

"""
    αβrecursion(cfsm, lhs)
    αβrecursion(π, ω, A, Aᵀ, lhs)

Return the states conditional probabilities (as a matrix) and the
total likelihood of the sequence. `csfm` is a compiled FSM
(see [`compile`](@ref)), `π` is the vector of initial probabilities,
`ω` is the vector of final probabilities, `A` is the matrix of
transition probabilites, `Aᵀ` is the transpose of the matrix of
transition probabilities and `lhs` is the matrix of likelihoods.
"""
αβrecursion

# A convenience function to call the forward-backward on a compiled
# FSM.
function αβrecursion(cfsm, lhs::AbstractArray)
    αβrecursion(cfsm.π, cfsm.ω, cfsm.A, cfsm.Aᵀ, lhs)
end

function αβrecursion(π::AbstractVector{T}, ω::AbstractVector{T},
                     A::AbstractMatrix{T}, Aᵀ::AbstractMatrix,
                     lhs::AbstractMatrix{T}) where T
    S, N = length(π), size(lhs, 2)
    α = similar(lhs, T, S, N)
    β = similar(lhs, T, S, N)

    αrecursion!(α, π, Aᵀ, lhs)
    βrecursion!(β, ω, A, lhs)

    γ = similar(lhs, T, S, N)
    elmul!(γ, α, β)
    sums = sum(γ, dims = 1)
    eldiv!(γ, γ, sums)

    γ, minimum(sums)
end

function αβrecursion(π::AbstractVector{T}, ω::AbstractVector{T},
                     A::AbstractMatrix{T}, Aᵀ::AbstractMatrix,
                     lhs::Abstract3DTensor{T}) where T
    S, N, B = size(lhs)
    α = similar(lhs, T, S, N, B)
    β = similar(lhs, T, S, N, B)

    αrecursion!(α, π, Aᵀ, lhs)
    βrecursion!(β, ω, A, lhs)

    γ = similar(lhs, T, S, N, B)
    elmul!(γ, α, β)
    sums = sum(γ, dims = 1)
    eldiv!(γ, γ, sums)

    γ, dropdims(minimum(sums, dims = (1, 2)), dims = (1, 2))
end

