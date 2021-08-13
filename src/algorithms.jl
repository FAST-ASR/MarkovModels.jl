# SPDX-License-Identifier: MIT

#======================================================================
Forward recursion
======================================================================#

function αrecursion(π::AbstractVector{SR},
                    Tᵀ::AbstractMatrix{SR},
                    lhs::AbstractMatrix{SR}) where SR <: Semiring
    S, N = length(π), size(lhs, 2)
    α = similar(lhs, SR, S, N)
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

function βrecursion(ω::AbstractVector{SR},
                    T::AbstractMatrix{SR},
                    lhs::AbstractMatrix{SR}) where SR <: Semiring
    S, N = length(ω), size(lhs, 2)
    β = similar(lhs, SR, S, N)
    buffer = similar(lhs[:, 1])

    @views elmul!(β[:, end], ω, fill!(similar(β[:,end]), one(SR)))
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

Calculate the conditional posterior of "assigning" the \$n\$th frame
to the \$i\$th pdf. The output is a tuple `γ, ttl` where `γ` is a matrix
(# pdf x # frames) and `ttl` is the total probability of the sequence.
This function can also be called in "batch-mode" by providing a union
of compiled fsm and a 3D tensor containing the per-state, per-frame
and per-batch values.
"""
pdfposteriors

function pdfposteriors(cfsm::CompiledFSM{SR},
                       in_lhs::AbstractMatrix{T}) where {SR<:LogSemifield,T}
    # Convert the likelihood matrix to the cfsm' semiring.
    lhs = copyto!(similar(in_lhs, SR), in_lhs)

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

function pdfposteriors(ucfsm::UnionCompiledFSM{SR},
                       in_lhs::AbstractArray{T,3}) where {SR<:LogSemifield,T}

    # Reshape the 3D tensor to have a matrix of size BK x N where
    # B is the number of elements in the batch.
    in_lhs_matrix = vcat(eachslice(in_lhs, dims = 3)...)

    # Compiled FSM which is set of parallel FSMs.
    cfsm = ucfsm.cfsm

    # Convert the likelihood matrix to the cfsm' semiring.
    lhs = copyto!(similar(in_lhs_matrix, SR), in_lhs_matrix)

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

"""
    maxstateposteriors(cfsm, lhs)

Calculate the posterior of "assigning" the \$n\$th frame
to the \$i\$th state conditioned of all other states maximizing the
likelihood of the sequence. The output is a matrix `μ` (# pdf x # frames).
"""
function maxstateposteriors(cfsm::CompiledFSM{SR},
                            in_lhs::AbstractArray) where SR<:TropicalSemiring
    # Convert the FSM and the likelihood matrix to the Log-semifield.
    lhs = copyto!(similar(in_lhs, SR), in_lhs)

    S = size(cfsm.C, 1)
    N = size(lhs, 2)

    # Expand the likelihood matrix to get the per-state likelihoods.
    state_lhs = matmul!(similar(lhs, S, N), cfsm.C, lhs)

    α = αrecursion(cfsm.π, cfsm.Tᵀ, state_lhs)
    β = βrecursion(cfsm.ω, cfsm.T, state_lhs)
    elmul!(similar(state_lhs), α, β)
end

function findmaxstate(cfsm::CompiledFSM{SR}, μ::AbstractArray{SR},
                      prevstate::Int) where {SR,T}
    m = typemin(SR)
    s = -1
    for (i, μᵢ) in pairs(μ)
        if μᵢ >= m && cfsm.T[prevstate,i] ≠ zero(SR)
            m = μᵢ
            s = i
        end
    end
    s
end

"""
    bestpath(cfsm, μ)

Return the sequence of state with the highest value from `μ`, i.e.
the ouput of [`maxstateposteriors`](@ref).
"""
function bestpath(cfsm::CompiledFSM{SR}, μ::AbstractArray{SR}) where SR
    seq = Vector{Int}(undef, size(μ, 2))
    seq[1] = argmax(μ[:,1])
    for n in 2:size(μ,2)
        seq[n] = findmaxstate(cfsm, μ[:,n], seq[n-1])
    end
    seq
end

