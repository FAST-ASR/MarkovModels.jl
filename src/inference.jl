# SPDX-License-Identifier: MIT

struct CompiledFSM{K}
    α̂
    T̂
    T̂ᵀ
    Ĉ
    Ĉᵀ
end

compile(fsm::FSM, Ĉ::AbstractMatrix) =
    CompiledFSM{eltype(fsm.α̂)}(fsm.α̂, fsm.T̂, copy(fsm.T̂'), Ĉ, copy(Ĉ'))

function Adapt.adapt_structure(::Type{<:CuArray}, cfsm::CompiledFSM{K}) where K
    T̂ = CuSparseMatrixCSC(cfsm.T̂)
    T̂ᵀ = CuSparseMatrixCSC(cfsm.T̂ᵀ)
    Ĉ = CuSparseMatrixCSC(cfsm.Ĉ)
    Ĉᵀ = CuSparseMatrixCSC(cfsm.Ĉᵀ)
    CompiledFSM{K}(
        CuSparseVector(cfsm.α̂),
        CuSparseMatrixCSR(T̂ᵀ.colPtr, T̂ᵀ.rowVal, T̂ᵀ.nzVal, T̂.dims),
        CuSparseMatrixCSR(T̂.colPtr, T̂.rowVal, T̂.nzVal, T̂ᵀ.dims),
        CuSparseMatrixCSR(Ĉᵀ.colPtr, Ĉᵀ.rowVal, Ĉᵀ.nzVal, Ĉ.dims),
        CuSparseMatrixCSR(Ĉ.colPtr, Ĉ.rowVal, Ĉ.nzVal, Ĉᵀ.dims),
    )
end

function batch(fsm1::CompiledFSM{K}, fsms::CompiledFSM{K}...) where K
    CompiledFSM{K}(
        vcat(fsm1.α̂, map(fsm -> fsm.α̂, fsms)...),
        blockdiag(fsm1.T̂, map(fsm -> fsm.T̂, fsms)...),
        blockdiag(fsm1.T̂ᵀ, map(fsm -> fsm.T̂ᵀ, fsms)...),
        blockdiag(fsm1.Ĉ, map(fsm -> fsm.Ĉ, fsms)...),
        blockdiag(fsm1.Ĉᵀ, map(fsm -> fsm.Ĉᵀ, fsms)...)
    )
end

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

function αrecursion(α̂::AbstractVector{K}, T̂ᵀ::AbstractMatrix{K},
                    lhs::AbstractMatrix{K}, seqlengths, cum_bsizes) where K
    S, N = length(α̂), size(lhs, 2)
    A = similar(lhs, K, S, N)
    buffer = similar(lhs[:, 1])

    @views broadcast!(*, A[:, 1], α̂, lhs[:, 1])
    @views for n in 2:N
        q = min(length(cum_bsizes), searchsortedlast(seqlengths, n-1; rev=true))
        q = max(1, q)
        mul!(buffer, T̂ᵀ, PartialVector(A[:, n - 1], cum_bsizes[q]))
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

function βrecursion(T̂::AbstractMatrix{K}, lhs::AbstractMatrix{K}, seqlengths, cum_bsizes) where K
    S, N = size(T̂, 1), size(lhs, 2)
    B = similar(lhs, K, S, N)
    buffer = similar(lhs[:, 1], S)

    @views fill!(B[:, end], one(K))
    @views for n in N-1:-1:1
        q = min(length(cum_bsizes), searchsortedlast(seqlengths, n+1; rev=true))
        q = max(1, q)
        #@show n+1, q, cum_bsizes[q]
        elmul!(*, buffer, PartialVector(B[:, n+1], cum_bsizes[q]), lhs[:, n+1])
        #broadcast!(*, buffer, B[:, n+1], lhs[:, n+1])
        mul!(B[:, n], T̂, PartialVector(buffer, cum_bsizes[q]))
    end
    B
end

# Do the backward recursion and multiply with the α immediately.
# This avoids to allocate a second array for the β messages.
function βrecursion_mulα!(α::AbstractMatrix{K}, T::AbstractMatrix{K},
                          lhs::AbstractMatrix{K}) where K <: Semiring
    S, N = size(T, 1), size(lhs, 2)
    βₘ = fill!(similar(lhs[:, 1], S), one(K))
    buffer = similar(lhs[:, 1], S)

    @views for n in N-1:-1:1
        broadcast!(*, buffer, βₘ, lhs[:, n+1])
        mul!(βₘ, T, buffer)
        broadcast!(*, α[:,n], α[:,n], βₘ)
    end
    α
end

function pdfposteriors(fsm::FSM{K}, V̂s, Ĉs) where K
    V̂ = vcat(V̂s...)
    V̂k = copyto!(similar(V̂, K), V̂)
    Ĉ = blockdiag(Ĉs...)
    Ĉᵀ = copy(Ĉ')
    ĈV̂ = (Ĉ * V̂k)
    T̂ᵀ = copy(fsm.T̂')
    state_A = αrecursion(fsm.α̂, T̂ᵀ, ĈV̂)
    state_B = βrecursion(fsm.T̂, ĈV̂)
    state_AB = state_A .* state_B
    AB = mul!(V̂k, Ĉᵀ, state_AB)
    Ẑ = permutedims(reshape(AB, :, length(V̂s), size(V̂, 2)), (2, 1, 3))
    sums = sum(Ẑ, dims = 2)
    Ẑ = broadcast!(/, Ẑ, Ẑ, sums)
    ttl = dropdims(minimum(sums, dims = (2, 3)), dims = (2, 3))
    (exp ∘ val).(Ẑ[:, 1:end-1, 1:end-1]), val.(ttl)
end


function pdfposteriors2(cfsm::CompiledFSM{K}, V̂s) where K
    V̂ = vcat(V̂s...)
    V̂k = copyto!(similar(V̂, K), V̂)
    ĈV̂ = (cfsm.Ĉ * V̂k)

    state_A = αrecursion(cfsm.α̂, cfsm.T̂ᵀ, ĈV̂)
    state_B = βrecursion(cfsm.T̂, ĈV̂)
    state_AB = state_A .* state_B
    AB = mul!(V̂k, cfsm.Ĉᵀ, state_AB)

    Ẑ = permutedims(reshape(AB, :, length(V̂s), size(V̂, 2)), (2, 1, 3))
    sums = sum(Ẑ, dims = 2)
    Ẑ = broadcast!(/, Ẑ, Ẑ, sums)
    ttl = dropdims(minimum(sums, dims = (2, 3)), dims = (2, 3))

    (exp ∘ val).(Ẑ[:, 1:end-1, 1:end-1]), val.(ttl)
end

function pdfposteriors3(cfsm::CompiledFSM{K}, V̂s, seqlengths, bsizes) where K
    cum_bsizes = [cumsum(bsizes)...]

    V̂ = vcat(V̂s...)
    V̂k = copyto!(similar(V̂, K), V̂)
    ĈV̂ = (cfsm.Ĉ * V̂k)

    state_A = αrecursion(cfsm.α̂, cfsm.T̂ᵀ, ĈV̂, seqlengths, cum_bsizes)
    #state_A = αrecursion(cfsm.α̂, cfsm.T̂ᵀ, ĈV̂)
    state_B = βrecursion(cfsm.T̂, ĈV̂, seqlengths, cum_bsizes)
    #state_B = βrecursion(cfsm.T̂, ĈV̂)
    state_AB = state_A .* state_B
    AB = mul!(V̂k, cfsm.Ĉᵀ, state_AB)

    Ẑ = permutedims(reshape(AB, :, length(V̂s), size(V̂, 2)), (2, 1, 3))
    sums = sum(Ẑ, dims = 2)
    Ẑ = broadcast!(Ẑ, Ẑ, sums) do x, y
        iszero(x) && iszero(y) ? zero(x) : x / y
    end
    ttl = sums[:, 1, 1]

    (exp ∘ val).(Ẑ[:, 1:end-1, 1:end-1]), val.(ttl)
end

