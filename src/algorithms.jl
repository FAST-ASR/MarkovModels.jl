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
        matmul!(buffer, Aᵀ, α[:,n-1,:])
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
Generic forward-backward algorithm
======================================================================#

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

#======================================================================
Specialized algorithms
======================================================================#

_convert(T, ttl::Number) = convert(T, ttl)
_convert(T, ttl::AbstractVector) = copyto!(similar(ttl, T), ttl)

function stateposteriors(in_cfsm, in_lhs::AbstractArray{T}) where T <: Real
    SF = LogSemifield{T}
    cfsm = convert(CompiledFSM{SF}, in_cfsm)
    lhs = copyto!(similar(in_lhs, SF), in_lhs)
    γ, ttl = αβrecursion(cfsm.π, cfsm.ω, cfsm.A, cfsm.Aᵀ, lhs)
    out = copyto!(similar(in_lhs), γ)
    exp.(out), _convert(T, ttl)
end

function bestpath(in_cfsm, in_lhs::AbstractArray{T}) where T <: Real
    SF = TropicalSemifield{T}
    cfsm = convert(CompiledFSM{SF}, in_cfsm)
    lhs = copyto!(similar(in_lhs, SF), in_lhs)
    γ, _ = αβrecursion(cfsm.π, cfsm.ω, cfsm.A, cfsm.Aᵀ, lhs)
    out = copyto!(similar(in_lhs), γ)
    idxs = mapslices(argmax, out, dims = 1)
    dropdims(idxs, dims = 1), dropdims(lhs[idxs], dims = 1)
end

