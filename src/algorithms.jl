# SPDX-License-Identifier: MIT

#======================================================================
Forward recursion
======================================================================#

function αrecursion(π::AbstractVector{SF},
                    Tᵀ::AbstractMatrix{SF},
                    lhs::AbstractMatrix{SF}) where SF <: Semifield
    S, N = length(π), size(lhs, 2)
    α = similar(lhs, SF, S, N)
    buffer = similar(lhs[:, 1])

    @views elmul!(α[:,1], π, lhs[:,1])
    @views for n in 2:N
        matmul!(buffer, Tᵀ, α[:,n-1])
        elmul!(α[:,n], buffer, lhs[:,n])
    end
    α
end

#======================================================================
Backward recursion
======================================================================#

function βrecursion(ω::AbstractVector{SF},
                    T::AbstractMatrix{SF},
                    lhs::AbstractMatrix{SF}) where SF <: Semifield
    S, N = length(ω), size(lhs, 2)
    β = similar(lhs, SF, S, N)
    buffer = similar(lhs[:, 1])

    β[:, end] = ω
    @views for n in N-1:-1:1
        elmul!(buffer, β[:,n+1], lhs[:,n+1])
        matmul!(β[:,n], T, buffer)
    end
    β
end

#======================================================================
Specialized algorithms
======================================================================#

"""
    pdfposteriors(cfsm, lhs)
    pdfposteriors(union(cfsm1, cfsm2, ...), batch_lhs)

Calculate the conditional posterior of "assigning" the $n$th frame
to the $i$th pdf. The output is a tuple `γ, ttl` where `γ` is a matrix
(# pdf x # frames) and `ttl` is the total probability of the sequence.
This function can also be caused in "batch-mode" by providing a union
of compiled fsm and a 3D tensor containing the per-state, per-frame
and per-batch values.
"""
pdfposteriors

function pdfposteriors(in_cfsm::CompiledFSM,
                       in_lhs::AbstractMatrix{T}) where T <: Real
    # Convert the FSM and the likelihood matrix to the Log-semifield.
    SF = LogSemifield{T}
    cfsm = convert(CompiledFSM{SF}, in_cfsm)
    lhs = copyto!(similar(in_lhs, SF), in_lhs)

    S = size(cfsm.C, 1)
    N = size(lhs, 2)

    # Expand the likelihood matrix to get the per-state likelihoods.
    state_lhs = matmul!(similar(lhs, S, N), cfsm.C, lhs)

    α = αrecursion(cfsm.π, cfsm.Tᵀ, state_lhs)
    β = βrecursion(cfsm.ω, cfsm.T, state_lhs)
    state_γ = elmul!(similar(state_lhs), α, β)

    # Transform the per-state γs to per-likelihoods γs.
    γ = matmul!(lhs, cfsm.Cᵀ, state_γ) # re-use `lhs` memory.

    # Re-normalize the γs.
    sums = sum(γ, dims = 1)
    eldiv!(γ, γ, sums)

    # Convert the result to the Real-semiring
    out = copyto!(similar(in_lhs), γ)

    exp.(out), convert(T, minimum(sums))
end

function pdfposteriors(in_ucfsm::UnionCompiledFSM,
                       in_lhs::AbstractArray{T,3}) where T <: Real

    # Reshape the 3D tensor to have a matrix of size BK x N where
    # B is the number of elements in the batch.
    in_lhs_matrix = vcat(eachslice(in_lhs, dims = 3)...)

    # Convert the FSM and the likelihood matrix to the Log-semifield.
    SF = LogSemifield{T}
    cfsm = convert(CompiledFSM{SF}, in_ucfsm.cfsm)
    lhs = copyto!(similar(in_lhs_matrix, SF), in_lhs_matrix)

    S = size(cfsm.C, 1)
    K, N = size(in_lhs)

    # Expand the likelihood matrix to get the per-state likelihoods.
    state_lhs = matmul!(similar(lhs, S, N), cfsm.C, lhs)

    α = αrecursion(cfsm.π, cfsm.Tᵀ, state_lhs)
    β = βrecursion(cfsm.ω, cfsm.T, state_lhs)
    state_γ = elmul!(similar(state_lhs), α, β)

    # Transform the per-state γs to per-likelihoods γs.
    γ = matmul!(lhs, cfsm.Cᵀ, state_γ) # re-use `lhs` memory.
    γ = permutedims(reshape(γ, K, :, N), (1, 3, 2))

    # Re-normalize each element of the batch.
    sums = sum(γ, dims = 1)
    eldiv!(γ, γ, sums)

    # Convert the result to the Real-semiring
    out = copyto!(similar(in_lhs), γ)

    ttl = dropdims(minimum(sums, dims = (1, 2)), dims = (1, 2))
    exp.(out), copyto!(similar(ttl, T), ttl)
end

@deprecate stateposteriors(cfsm, lhs) pdfposteriors(cfsm, lhs)

function bestpath(in_cfsm, in_lhs::AbstractArray{T}) where T <: Real
    SF = TropicalSemifield{T}
    cfsm = convert(CompiledFSM{SF}, in_cfsm)
    lhs = copyto!(similar(in_lhs, SF), in_lhs)
    γ, _ = αβrecursion(cfsm.π, cfsm.ω, cfsm.A, cfsm.Aᵀ, lhs)
    out = copyto!(similar(in_lhs), γ)
    idxs = mapslices(argmax, out, dims = 1)
    dropdims(idxs, dims = 1), dropdims(lhs[idxs], dims = 1)
end

