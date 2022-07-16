# SPDX-License-Identifier: MIT

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
    rawunion(fsms::FSM{K}...) where K

Contrary to the standard union, the raw union blindly stack the
internal storages of the FSMs. Consequently, the "virtual" final state
won't be merge together and the resulting FSM will have several
"virtual" final state. The output of `rawunion` should be considered
as several independent FSMs packed in a single structure.
"""
function rawunion(fsm1::FSM{K}, fsm2::FSM{K}) where K
    FSM(
        vcat(fsm1.α̂, fsm2.α̂),
        blockdiag(fsm1.T̂, fsm2.T̂),
        vcat(fsm1.λ, fsm2.λ)
    )
end
rawunion(fsm1::FSM{K}, fsms::FSM{K}...) where K =
    foldl(rawunion, fsms, init = fsm1)


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
        vcat([λ₁ᵢ * fsmⁱ.λ for (λ₁ᵢ, fsmⁱ) in zip(fsm₁.λ, fsms)]...)
    )
end


function compose(fsm1::FSM, dictfsms::AbstractDict)
    compose(fsm1, [dictfsms[Label(val(λᵢ)[end])]  for λᵢ in fsm1.λ])
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
        #SparseArrays.fkeep!(v, (i, x) -> i ∉ visited)
        #if nnz(v) > 0 push!(visited, findnz(v)[1]...) end
    end
    FSM(fsm.α, A, o, fsm.λ)
end


# Extract the non-zero states as an array of tuples.
_det_getstates(x) = [tuple(sort(map(a -> val(a)[1], collect(val(x))))...)
                     for x in nonzeros(x)]

"""
    determinize(fsm[, match])

Return an equivalent deterministic FSM. States `i` and `j` can be
merged if `match(i, j)` is `true`. Note that to guarantee the
equivalence of the returned FSM, you need to [`propagate`](@ref) weight
on `fsm` prior to call `determinize`.
"""
function determinize(fsm::FSM{K}, match = Base.:(==)) where K
    # We precompute the necessary matrices to estimate the new states
    # (i.e. set of states of the original fsm) and their transition weight.
    labels = sort(collect(Set(fsm.λ)), by = val)
    Mₗ = mapping(UnionConcatSemiring{LabelMonoid}, 1:nstates(fsm), labels,
                 (i, l) -> fsm.λ[i] == l)
    M = mapping(K, 1:nstates(fsm), labels, (i, l) -> fsm.λ[i] == l)
    α, T, ω = fsm.α, fsm.T, fsm.ω
    statelabels = UnionConcatSemiring{LabelMonoid}.([Set([Label(i)]) for i in 1:nstates(fsm)])
    αₗ = tobinary(UnionConcatSemiring{LabelMonoid}, fsm.α) .* statelabels
    Tₗ = tobinary(UnionConcatSemiring{LabelMonoid}, fsm.T) * spdiagm(statelabels)

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

        zₗ = sparsevec(collect(state), one(UnionConcatSemiring{LabelMonoid}),
                       nstates(fsm))
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

