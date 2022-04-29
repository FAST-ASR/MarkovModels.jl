# SPDX-License-Identifier: MIT

#======================================================================
Total sum
======================================================================#

function totalsum(α, T, ω, n)
    v = α
    total = dot(v, ω)
    for i in 2:n
        v = T' * v
        total += dot(v, ω)
    end
    total
end

totalweightsum(fsm::FSM, n) = totalsum(fsm.α, fsm.T, fsm.ω, n)

totallabelsum(fsm::FSM, n) = totalsum(tobinary(UnionConcatSemiring, fsm.α),
                                      tobinary(UnionConcatSemiring, fsm.T),
                                      tobinary(UnionConcatSemiring, fsm.ω),
                                      n)

#======================================================================
Union
======================================================================#

function Base.union(fsm1::FSM{K}, fsm2::FSM{K}) where K
    FSM(
        vcat(fsm1.α, fsm2.α),
        blockdiag(fsm1.T, fsm2.T),
        vcat(fsm1.ω, fsm2.ω),
        vcat(fsm1.λ, fsm2.λ)
    )
end

#======================================================================
Concatenation
======================================================================#

function Base.cat(fsm1::FSM{K}, fsm2::FSM{K}) where K
    FSM(
        vcat(fsm1.α, zero(fsm2.α)),
        [fsm1.T       fsm1.ω * fsm2.α';
         zero(fsm2.ω * fsm1.α')     fsm2.T],
        vcat(zero(fsm1.ω), fsm2.ω),
        vcat(fsm1.λ, fsm2.λ)
    )
end

#======================================================================
Reversal (a.k.a. FSM transposition)
======================================================================#

function Base.adjoint(fsm::FSM)
    FSM(fsm.ω, fsm.T', fsm.α, fsm.λ)
end

#======================================================================
Renormalization
======================================================================#

function renorm(fsm::FSM{K}) where K
    Z = one(K) ./ (sum(fsm.T, dims=2) .+ fsm.ω)
    FSM(
        fsm.α ./ sum(fsm.α),
        fsm.T .* Z,
        fsm.ω .* Z[:,1],
        fsm.λ
    )
end

#======================================================================
Composition
======================================================================#

function _mapping_matrix(K::Type{<:Semiring}, fsm₁, fsms)
    blocks = []
    for fsmⁱ in fsms
        n = nstates(fsmⁱ)
        push!(blocks, sparse(1:n, ones(n), ones(K, n)))
    end
    blockdiag(blocks...)
end

function _weighted_sparse_vcat(x, ys)
    K = eltype(x)
    I, V = Int64[], K[]
    count = 0
    for i in 1:length(x)
        n = length(ys[i])
        if ! iszero(x[i])
            J, W = findnz(ys[i])
            append!(I, J .+ count)
            append!(V, x[i] * W)
        end
        count += n
    end
    return sparsevec(I, V, count)
end

function compose(fsm₁::FSM, fsms::AbstractVector{<:FSM{K}},
                 sep = one(UnionConcatSemiring)) where K
    ω = vcat([fsmⁱ.ω for fsmⁱ in fsms]...)
    α = vcat([fsmⁱ.α for fsmⁱ in fsms]...)
    Mₖ = _mapping_matrix(K, fsm₁, fsms)
    T₂ = blockdiag([fsmⁱ.T for fsmⁱ in fsms]...) + (Mₖ * fsm₁.T * Mₖ') .* (ω * α')

    FSM(
        _weighted_sparse_vcat(fsm₁.α, [fsmⁱ.α for fsmⁱ in fsms]),
        T₂,
        _weighted_sparse_vcat(fsm₁.ω, [fsmⁱ.ω for fsmⁱ in fsms]),
        vcat([λ₁ᵢ * sep * fsmⁱ.λ for (λ₁ᵢ, fsmⁱ) in zip(fsm₁.λ, fsms)]...)
    )
end
Base.:∘(fsm₁::FSM, fsms::AbstractVector) = compose(fsm₁, fsms)

#======================================================================
Determinization
======================================================================#

function _det_label_matrices(fsm)
    I, J, V = [], [], UnionConcatSemiring[]
    for i in 1:nstates(fsm), j in 1:nstates(fsm)
        if ! iszero(fsm.T[i, j])
            push!(I, i)
            push!(J, j)
            push!(V, Label(j))
        end
    end
    Tₗ = sparse(I, J, V, nstates(fsm), nstates(fsm))

    I, _ = findnz(fsm.α)
    αₗ = sparsevec(I, Label.(I), nstates(fsm))

    αₗ, Tₗ
end

function _det_mapping(fsm, match)
    labels = [Label(l...) for l in sort(collect(val(sum(fsm.λ))))]

    I, J, V = [], [], UnionConcatSemiring[]
    for i in 1:nstates(fsm), j in 1:length(labels)
        if match(fsm.λ[i], labels[j])
            push!(I, i)
            push!(J, j)
            push!(V, one(UnionConcatSemiring))
        end
    end
    Mₗ = sparse(I, J, V, nstates(fsm), length(labels))

    K = eltype(fsm.α)
    I, J, V = [], [], K[]
    for i in 1:nstates(fsm), j in 1:length(labels)
        if match(fsm.λ[i], labels[j])
            push!(I, i)
            push!(J, j)
            push!(V, one(K))
        end
    end
    M = sparse(I, J, V, nstates(fsm), length(labels))

    labels, Mₗ, M
end

_det_getstates(x) = [tuple(sort(map(first, collect(val(x))))...)
                     for x in nonzeros(x)]

function determinize(fsm::FSM{K}, match = Base.:(==)) where K
    labels, Mₗ, M = _det_mapping(fsm, match)
    αₗ, Tₗ = _det_label_matrices(fsm)
    α, T, ω = fsm.α, fsm.T, fsm.ω

    states = _det_getstates(Mₗ' * αₗ)
end
