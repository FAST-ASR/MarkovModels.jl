# SPDX-License-Identifier: MIT


function totalsum(α, T, ω, n)
    v = α
    total = dot(v, ω)
    for i in 2:n
        v = T' * v
        total += dot(v, ω)
    end
    total
end

"""
    totalweightsum(fsm::FSM, n)

Compute the `n`th partial total weight sum of `fsm`.
"""
totalweightsum(fsm::FSM, n = nstates(fsm)) = totalsum(fsm.α, fsm.T, fsm.ω, n)

"""
    totalweightsum(fsm::FSM, n)

Compute the `n`th partial total label sum of `fsm`.
"""
function totallabelsum(fsm::FSM, n = nstates(fsm))
    λ = UnionConcatSemiring.([Set([λᵢ]) for λᵢ in fsm.λ])
    totalsum(
        tobinary(UnionConcatSemiring, fsm.α) .* λ,
        tobinary(UnionConcatSemiring, fsm.T) * spdiagm(λ),
        tobinary(UnionConcatSemiring, fsm.ω),
        n
   )
end

"""
    union(fsms::FSM{K}...) where K

Return the union of the given FSMs.
"""
function Base.union(fsm1::FSM{K}, fsm2::FSM{K}) where K
    FSM(
        vcat(fsm1.α, fsm2.α),
        blockdiag(fsm1.T, fsm2.T),
        vcat(fsm1.ω, fsm2.ω),
        vcat(fsm1.λ, fsm2.λ)
    )
end
Base.union(fsm1::FSM{K}, fsms::FSM{K}...) where K =
    foldl(union, fsms, init = fsm1)


"""
    cat(fsms::FSM{K}...) where K

Return the concatenated FSMs.
"""
function Base.cat(fsm1::FSM{K}, fsm2::FSM{K}) where K
    FSM(
        vcat(fsm1.α, zero(fsm2.α)),
        [fsm1.T       fsm1.ω * fsm2.α';
         zero(fsm2.ω * fsm1.α')     fsm2.T],
        vcat(zero(fsm1.ω), fsm2.ω),
        vcat(fsm1.λ, fsm2.λ)
    )
end
Base.cat(fsm1::FSM{K}, fsms::FSM{K}...) where K =
    foldl(cat, fsms, init = fsm1)

"""
    Base.adjoint(fsm::FSM)
    fsm'

Return the reversal of `fsm`.
"""
function Base.adjoint(fsm::FSM)
    FSM(fsm.ω, sparse(fsm.T'), fsm.α, fsm.λ)
end

"""
    renorm(fsm::FSM)

Return a normalized FSM.
"""
function renorm(::Type{Divisible}, fsm::FSM{K}) where K
    Z = one(K) ./ (sum(fsm.T, dims=2) .+ fsm.ω)
    FSM(
        fsm.α ./ sum(fsm.α),
        fsm.T .* Z,
        fsm.ω .* Z[:,1],
        fsm.λ
    )
end
renorm(fsm::FSM{K}) where K = renorm(IsDivisible(K), fsm)

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

"""
    compose(fsm₁, fsms)

Return the composition of `fsm₁` with a list of FSMs.
"""
function compose(fsm₁::FSM, fsms::AbstractVector{<:FSM{K}},
                 sep = Label(":")) where K
    A = blockdiag([fsmⁱ.α[:,1:1] for fsmⁱ in fsms]...)
    Ω = blockdiag([fsmⁱ.ω[:,1:1] for fsmⁱ in fsms]...)

    FSM(
        _weighted_sparse_vcat(fsm₁.α, [fsmⁱ.α for fsmⁱ in fsms]),
        blockdiag([fsmⁱ.T for fsmⁱ in fsms]...) + Ω * fsm₁.T * A',
        _weighted_sparse_vcat(fsm₁.ω, [fsmⁱ.ω for fsmⁱ in fsms]),
        vcat([compose.(λ₁ᵢ, fsmⁱ.λ) for (λ₁ᵢ, fsmⁱ) in zip(fsm₁.λ, fsms)]...)
    )
end


function compose(fsm1::FSM, dictfsms::AbstractDict)
    compose(fsm1, [dictfsms[decompose(λᵢ)[end]]  for λᵢ in fsm1.λ])
end

Base.:∘(fsm1::FSM, fsm2) = compose(fsm1, fsm2)

"""
    propagate(fsm)

Propagate the weights along the FSM's arcs.
"""
function propagate(fsm::FSM{K}) where K
    v = fsm.α
    A = spdiagm(v) * fsm.T
    o = fsm.ω .* v#spzeros(K, nstates(fsm))
    visited = Set(findnz(v)[1])
    for n in 2:(nstates(fsm))
        v = fsm.T' * v
        A += spdiagm(v) * fsm.T
        o += fsm.ω .* v

        # Prune the states that have been visited.
        SparseArrays.fkeep!(v, (i, x) -> i ∉ visited)
        if nnz(v) > 0 push!(visited, findnz(v)[1]...) end
    end
    FSM(fsm.α, A, o, fsm.λ)
end


# Extract the non-zero states as an array of tuples.
_det_getstates(x) = [tuple(sort(map(a -> parse(Int, a), collect(val(x))))...)
                     for x in nonzeros(x)]

"""
    determinize(fsm[, match])
Return an equivalent deterministic FSM. States `i` and `j` can be
merged if `match(i, j)` is `true`. Note that to guarantee the
equivalence of the returned FSM, you need to [`propagate`](@ref) weight
on `fsm` prior to call `determinize`.
"""
function determinize(fsm::FSM{K}, match = Base.:(==)) where K
    λ = UnionConcatSemiring.([Set([λᵢ]) for λᵢ in fsm.λ])

    # We precompute the necessary matrices to estimate the new states
    # (i.e. set of states of the original fsm) and their transition weight.
    labels = [UnionConcatSemiring(Set([l])) for l in sort(collect(val(sum(λ))))]
    Mₗ = mapping(UnionConcatSemiring, 1:nstates(fsm), labels, (i, l) -> λ[i] == l)
    M = mapping(K, 1:nstates(fsm), labels, (i, l) -> λ[i] == l)
    α, T, ω = fsm.α, fsm.T, fsm.ω
    statelabels = UnionConcatSemiring.([Set(["$i"]) for i in 1:nstates(fsm)])
    αₗ = tobinary(UnionConcatSemiring, fsm.α) .* statelabels
    Tₗ = tobinary(UnionConcatSemiring, fsm.T) * spdiagm(statelabels)

    # Initialize the queue for the powerset construction algorithm.
    newstates = Dict()
    newarcs = Dict()
    initstates = _det_getstates(Mₗ' * αₗ)
    queue = Array{Tuple}(initstates)
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
    FSM(α₂, T₂, ω₂, newlabels)
end

"""
    minimize(fsm[, match])

Return an equivalent minimal FSM. Note that to guarantee the
equivalence of the returned FSM, you need to [`propagate`](@ref) weight
on `fsm` prior to call `minimize`.
"""
minimize = adjoint ∘ determinize ∘ adjoint ∘ determinize

