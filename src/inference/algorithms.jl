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

function βrecursion(T::AbstractMatrix{SR},
                    lhs::AbstractMatrix{SR}) where SR <: Semiring
    S, N = size(T, 1), size(lhs, 2)
    β = fill!(similar(lhs, SR, S, N), one(SR))
    buffer = similar(lhs[:, 1], S)

    @views for n in N-1:-1:1
        elmul!(buffer, β[:,n+1], lhs[:,n+1])
        matmul!(β[:,n], T, buffer)
    end
    β
end

#======================================================================
Specialized algorithms
======================================================================#

function _expand(lhs::AbstractMatrix{T}, seqlength = size(lhs, 2)) where T
    S, N = size(lhs)
    retval = hcat(vcat(lhs, zeros(T, 1, N)), zeros(T, S+1))
    @views fill!(retval[1:end-1,seqlength+1:end], zero(T))
    @views fill!(retval[end,seqlength+1:end], one(T))
    retval
end

function _split(fsm, lhs)
	offset = 0
	retval = []
	for r in fsm.ranges
		nr = (r[1]:r[end]-1) .- offset
		push!(retval, lhs[nr, :])
		offset += 1
	end
	retval
end

function _drop_extradims(fsm, γ)
    [γ[r[1]:r[end]-1, 1:end-1] for r in fsm.ranges]
end

"""
    pdfposteriors(mfsm, lhs)
    pdfposteriors(union(mfsm1, mfsm2, ...), batch_lhs)

Calculate the conditional posterior of "assigning" the \$n\$th frame
to the \$i\$th pdf. The output is a tuple `γ, ttl` where `γ` is a matrix
(# pdf x # frames) and `ttl` is the total probability of the sequence.
This function can also be called in "batch-mode" by providing a union
of compiled fsm and a 3D tensor containing the per-state, per-frame
and per-batch values.
"""
pdfposteriors

function pdfposteriors(mfsm::MatrixFSM{SR},
                       in_lhs::AbstractMatrix{T}) where {SR<:LogSemifield,T}
    # Convert the likelihood matrix to the mfsm' semiring.
    lhs = copyto!(similar(in_lhs, SR), in_lhs)

    # Add one frame and one dimension for the final state.
    expanded_lhs = _expand(lhs)

    S = size(mfsm.C, 1)
    N = size(expanded_lhs, 2)

    # Get the per-state likelihood matrix.
    state_lhs = matmul!(similar(lhs, S, N), mfsm.C, expanded_lhs)

    α = αrecursion(mfsm.π, mfsm.Tᵀ, state_lhs)
    β = βrecursion(mfsm.T, state_lhs)
    state_γ = elmul!(similar(state_lhs), α, β)

    # Transform the per-state γs to per-likelihoods γs.
    γ = matmul!(expanded_lhs, mfsm.Cᵀ, state_γ) # re-use `expanded_lhs` memory.

    # Re-normalize the γs.
    sums = sum(γ, dims = 1)
    eldiv!(γ, γ, sums)

    # Convert the result to the Real-semiring
    out = copyto!(similar(in_lhs), γ[1:end-1,1:end-1])

    exp.(out), convert(T, minimum(sums))
end

function pdfposteriors(mfsm::UnionMatrixFSM{SR},
                       in_lhs::AbstractArray{T,3},
                       seqlengths,
                      ) where {SR<:LogSemifield,T}

    # Convert the likelihood tensor to the mfsm's semiring.
    lhs_tensor = copyto!(similar(in_lhs, SR), in_lhs)

    # Reshape the 3D tensor to have a matrix of size BK x N where
    # B is the number of elements in the batch.
    lhs = vcat(
        map(t -> _expand(t...),
            zip(eachslice(lhs_tensor, dims = 3), seqlengths))...
    )

    S = size(mfsm.C, 1)
    K = size(in_lhs, 1) + 1
    N = size(in_lhs, 2) + 1

    # Get the per-state likelihoods.
    state_lhs = matmul!(similar(lhs, S, N), mfsm.C, lhs)

    α = αrecursion(mfsm.π, mfsm.Tᵀ, state_lhs)
    β = βrecursion( mfsm.T, state_lhs)
    state_γ = elmul!(similar(state_lhs), α, β)

    # Drop the last frame and the last dimension.
    #state_γ = _drop_extradims(mfsm, expanded_state_γ[1:end-1,1:end-1])

    # Transform the per-state γs to per-likelihoods γs.
    γ = matmul!(lhs, mfsm.Cᵀ, state_γ) # re-use `lhs` memory.
    γ = permutedims(reshape(γ, K, :, N), (1, 3, 2))

    # TODO before droping dimensions !!
    # Re-normalize each element of the batch.
    sums = sum(γ, dims = 1)
    eldiv!(γ, γ, sums)

    # Convert the result to the Real-semiring
    out = copyto!(similar(in_lhs), γ[1:end-1,1:end-1,:])

    ttl = dropdims(minimum(sums, dims = (1, 2)), dims = (1, 2))
    exp.(out), copyto!(similar(ttl, T), ttl)
end

@deprecate stateposteriors(mfsm, lhs) pdfposteriors(mfsm, lhs)


"""
    maxstateposteriors(mfsm, lhs)

Calculate the posterior of "assigning" the \$n\$th frame
to the \$i\$th state conditioned of all other states maximizing the
likelihood of the sequence. The output is a matrix `μ` (# pdf x # frames).
"""
function maxstateposteriors(mfsm::MatrixFSM{SR},
                            in_lhs::AbstractArray) where SR<:TropicalSemiring
    # Convert the FSM and the likelihood matrix to the Log-semifield.
    lhs = copyto!(similar(in_lhs, SR), in_lhs)

    # Add one frame and one dimension for the final state.
    expanded_lhs = _expand(lhs)

    S = size(mfsm.C, 1)
    N = size(expanded_lhs, 2)

    # Expand the likelihood matrix to get the per-state likelihoods.
    state_lhs = matmul!(similar(lhs, S, N), mfsm.C, expanded_lhs)

    α = αrecursion(mfsm.π, mfsm.Tᵀ, state_lhs)
    β = βrecursion(mfsm.T, state_lhs)
    elmul!(similar(state_lhs), α, β)[1:end-1,1:end-1]
end

function findmaxstate(mfsm::MatrixFSM{SR}, μ::AbstractArray{SR},
                      prevstate::Int) where {SR,T}
    m = typemin(SR)
    s = -1
    for (i, μᵢ) in pairs(μ)
        if μᵢ >= m && mfsm.T[prevstate,i] ≠ zero(SR)
            m = μᵢ
            s = i
        end
    end
    s
end

"""
    bestpath(mfsm, μ)

Return the sequence of state with the highest value from `μ`, i.e.
the ouput of [`maxstateposteriors`](@ref).
"""
function bestpath(mfsm::MatrixFSM{SR}, μ::AbstractArray{SR}) where SR
    seq = Vector{Int}(undef, size(μ, 2))
    seq[1] = argmax(μ[:,1])
    for n in 2:size(μ,2)
        seq[n] = findmaxstate(mfsm, μ[:,n], seq[n-1])
    end
    seq
end

