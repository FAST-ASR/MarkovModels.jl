# SPDX-License-Identifier: MIT

#======================================================================
We use a different data structure to store the FSA for inference.
======================================================================#
struct CompiledFSA{K, TT̂<:TransitionMatrix{K}, Tα̂<:WeightVector{K}}
    α̂::Tα̂
    T̂::TT̂
end

function compile(fsa::FSA)
    Tω = hcat(fsa.T, fsa.ω)
    p = fill!(similar(fsa.ω, nstates(fsa) + 1), zero(eltype(fsa.ω)))
    p[end] = one(eltype(fsa.ω))
    T̂ = vcat(Tω, reshape(p, 1, :))

    CompiledFSA(vcat(fsa.α, zero(eltype(fsa.α))), T̂)
end


Adapt.adapt_structure(::Type{<:CuArray}, fsa::CompiledFSA) =
    CompiledFSA(
        CuSparseVector(fsa.α̂),
        CuSparseMatrixCSR(CuSparseMatrixCSC(fsa.T̂)),
    )

"""
    batch(fsas::CompiledFSA{K}...) where K

Stack the internal storages of the FSAs. Consequently, the "virtual"
final state won't be merge together and the resulting FSA will have
several "virtual" final state. The output of `batch` should be
considered as several independent FSAs packed in a single structure.
"""
batch(fsa1::CompiledFSA{K}, fsa2::CompiledFSA{K}) where K =
    CompiledFSA(vcat(fsa1.α̂, fsa2.α̂), blockdiag(fsa1.T̂, fsa2.T̂))
batch(fsa1::CompiledFSA{K}, fsas::CompiledFSA{K}...) where K =
    foldl(batch, fsas, init = fsa1)

"""
    expand(V::AbstractMatrix{K}, seqlength = size(lhs, 2)) where K

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

function αrecursion(prune!::Function, α̂::AbstractVector{K},
                    T̂ᵀ::AbstractMatrix{K}, lhs::AbstractMatrix{K}) where K
    S, N = length(α̂), size(lhs, 2)
    A = similar(lhs, K, S, N)
    A = [prune!(lhs[:, 1] .* α̂)]

    for n in 2:N
        buffer = T̂ᵀ * A[n-1][
        push!(A, prune!(lhs[:, n] .* buffer))
    end
    LILMatrix(length(α̂), A)
end

αrecursion(α̂::AbstractVector{K}, T̂ᵀ::CuSparseAdjOrTrans{K},
           lhs::AbstractMatrix{K}) where K =
    αrecursion(α̂, copy(T̂ᵀ), lhs)

βrecursion(T̂ᵀ::CuSparseAdjOrTrans{K}, lhs::AbstractMatrix{K}) where K =
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

function pdfposteriors(fsm::CompiledFSA{K}, V̂s, Ĉs) where K
    V̂ = vcat(V̂s...)
    V̂k = copyto!(similar(V̂, K), V̂)
    Ĉ = blockdiag(Ĉs...)
    ĈV̂ = (Ĉ * V̂k)
    state_A = αrecursion(fsm.α̂, fsm.T̂', ĈV̂)
    state_B = βrecursion(fsm.T̂, ĈV̂)
    state_AB = broadcast!(*, state_A, state_A, state_B)
    AB = Ĉ' * state_AB
    Ẑ = permutedims(reshape(AB, :, length(V̂s), size(V̂, 2)), (2, 1, 3))
    sums = sum(Ẑ, dims = 2)
    Ẑ = broadcast!(/, Ẑ, Ẑ, sums)
    ttl = dropdims(minimum(sums, dims = (2, 3)), dims = (2, 3))
    (exp ∘ val).(Ẑ[:, 1:end-1, 1:end-1]), val.(ttl)
end

