# SPDX-License-Identifier: MIT

"""
    expand(V::AbstractMatrix{K}, seqlength = size(V, 2)) where K

Expand the ``D x N`` matrix of likelihoods `V` to a ``D+1 x N+1``
matrix `V̂`. This function is to prepare the matrix of likelihood to
be used for the forward-backward algorithm. The option `seqlength` is
mostly used for preparing batch calculation of the forward-bacward
algorithm.

The added extra row of `V̂` corresponds to the likelihood of a phony
final state. The resulting matrix `V̂` has the folloing values:
- `V̂[i, j] = V[i, j]` for `i ≤ D` and `j ≤ seqlength`
- `V̂[i, j] = zero(K)` for `i ≤ D` and `j > seqlength`
- `V̂[i, j] = zero(K)` for `i = D+1` and `j <= seqlength`
- `V̂[i, j] = one(K)` for `i = D+1` and `j > seqlength`.
"""
function expand(lhs::AbstractMatrix{K}, seqlength = size(lhs, 2)) where K
    S, N = size(lhs)
    retval = hcat(vcat(lhs, zeros(K, 1, N)), zeros(K, S+1))
    @views fill!(retval[1:end-1, seqlength+1:end], zero(K))
    @views fill!(retval[end, seqlength+1:end], one(K))
    retval
end

function αrecursion(α̂::AbstractVector{K}, T̂ᵀ::AbstractMatrix{K},
                    lhs::AbstractMatrix{K}) where K
    S, N = length(α̂), size(lhs, 2)
    A = similar(lhs, K, S, N)
    buffer = similar(lhs[:, 1])

    @views broadcast!(*, A[:, 1], α̂, lhs[:, 1])
    @views for n in 2:N
        mul!(buffer, T̂ᵀ, A[:, n - 1])
        broadcast!(*, A[:, n], buffer, lhs[:, n])
    end
    A
end

αrecursion(α̂::AbstractVector{K}, T̂ᵀ::CuAdjOrTranspose{K},
           lhs::AbstractMatrix{K}) where K =
    αrecursion(α̂, copy(T̂ᵀ), lhs)

βrecursion(T̂ᵀ::CuAdjOrTranspose{K}, lhs::AbstractMatrix{K}) where K =
    βrecursion(copy(T̂ᵀ), lhs)

function βrecursion(T̂::AbstractMatrix{K}, lhs::AbstractMatrix{K}) where K
    S, N = size(T̂, 1), size(lhs, 2)
    B = similar(lhs, K, S, N)
    buffer = similar(lhs[:, 1], S)

    @views fill!(B[:, end], one(K))
    @views for n in N-1:-1:1
        broadcast!(*, buffer, B[:, n+1], lhs[:, n+1])
        mul!(B[:, n], T̂, buffer)
    end
    B
end

"""
    pdfposteriors(fsm::FSM{K}, V̂s, Ĉs) where K

Compute pdf posteriors.

args:
    fsm - batched FSM
    V̂s - vector of pdf likelihoods of shape (n_pdfs x N)
    Ĉs - vector of state-pdf mappings
"""
function pdfposteriors(fsm::FSM{K}, V̂s, Ĉs) where K
    V̂ = vcat(V̂s...)  # B*n_pdfs x N
    V̂k = copyto!(similar(V̂, K), V̂)
    Ĉ = blockdiag(Ĉs...)  # B*n_states x B*n_pdfs
    ĈV̂ = (Ĉ * V̂k)  # B*n_states x N
    state_A = αrecursion(fsm.α̂, fsm.T̂', ĈV̂)
    state_B = βrecursion(fsm.T̂, ĈV̂)
    state_AB = broadcast!(*, state_A, state_A, state_B)  # B*n_states x N
    AB = Ĉ' * state_AB  # B*n_pdfs x N
    Ẑ = permutedims(reshape(AB, :, length(V̂s), size(V̂, 2)), (2, 1, 3)) # B x n_pdfs x N
    sums = sum(Ẑ, dims = 2)  # B x 1 x N
    Ẑ = broadcast!(/, Ẑ, Ẑ, sums)
    ttl = dropdims(minimum(sums, dims = (2, 3)), dims = (2, 3))
    (exp ∘ val).(Ẑ[:, 1:end-1, 1:end-1]), val.(ttl)
end

