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
totallabelsum(fsm::FSM, n) = totalsum(tobinary(UnionConcatSemiring, fsm.α) .* fsm.λ,
                                      tobinary(UnionConcatSemiring, fsm.T) * spdiagm(fsm.λ),
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
    FSM(fsm.ω, sparse(fsm.T'), fsm.α, fsm.λ)
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

# Extract the non-zero states as an array of tuples.
_det_getstates(x) = [tuple(sort(map(first, collect(val(x))))...)
                     for x in nonzeros(x)]

function determinize(fsm::FSM{K}, match = Base.:(==)) where K
    # We precompute the necessary matrices to estimate the new states
    # (i.e. set of states of the original fsm) and their transition weight.
    labels = [Label(l...) for l in sort(collect(val(sum(fsm.λ))))]
    Mₗ = mapping(UnionConcatSemiring, 1:nstates(fsm), labels, (i, l) -> fsm.λ[i] == l)
    M = mapping(K, 1:nstates(fsm), labels, (i, l) -> fsm.λ[i] == l)
    α, T, ω = fsm.α, fsm.T, fsm.ω
    statelabels = Label.(collect(1:nstates(fsm)))
    αₗ = tobinary(UnionConcatSemiring, fsm.α) .* statelabels
    Tₗ = tobinary(UnionConcatSemiring, fsm.T) * spdiagm(statelabels)

    # Initialize the queue for the powerset construction algorithm.
    newstates = Dict()
    newarcs = Dict()
    queue = Array{Tuple}(_det_getstates(Mₗ' * αₗ))
    for s in queue
        newstates[s] = (sum(α[collect(s)]), sum(ω[collect(s)]))
    end

    # Powerset construction algorithm.
    while ! isempty(queue)
        state = popfirst!(queue)
        finalweight = sum(ω[collect(state)])

        zₗ = sparsevec(collect(state), one(UnionConcatSemiring), nstates(fsm))
        nextstates = _det_getstates(Mₗ' * Tₗ' * zₗ)

        z = sparsevec(collect(state), one(K), nstates(fsm))
        nextweights = nonzeros(M' * T' * z)

        for (ns, nw) in zip(nextstates, nextweights)
            arcs = get(newarcs, state, [])
            push!(arcs, (ns, nw))
            newarcs[state] = arcs
            if ns ∉ keys(newstates)
                newstates[ns] = (zero(K), sum(ω[collect(ns)]))
                push!(queue, ns)
            end
        end
    end

    # We build the actual fsm from the result of the powerset
    # construction.
    state2idx = Dict(s => i for (i, s) in enumerate(keys(newstates)))
    newlabels = [fsm.λ[s[1]] for s in keys(newstates)]
    α₂ = []
    ω₂ = []
    T₂ = []
    for (i, s) in enumerate(keys(newstates))
        iw, fw = newstates[s]
        if ! iszero(iw) push!(α₂, i => iw) end
        if ! iszero(fw) push!(ω₂, i => fw) end
        if s in keys(newarcs)
            for (ns, nw) in newarcs[s]
                push!(T₂, (state2idx[s], state2idx[ns]) => nw)
            end
        end
    end
    FSM(length(newstates), α₂, T₂, ω₂, newlabels)
end

