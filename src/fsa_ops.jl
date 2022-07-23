# SPDX-License-Identifier: MIT

#=====================================================================#
# Utility functions

"""
    tobinary(K::Type{<:Semiring}, x)

Create `y`, a sparse vector or a sparse matrix of the same size of `x`
where `y[i] = one(K)` if `x[i] ≂̸ zero(K)`.
"""
tobinary(K::Type{<:Semiring}, x::AbstractSparseVector{<:Semiring}) =
    sparsevec(findnz(x)[1], ones(K, nnz(x)), length(x))
tobinary(K::Type{<:Semiring}, x::AbstractSparseMatrix{<:Semiring}) =
    sparse(findnz(x)[1], findnz(x)[2], ones(K, nnz(x)), size(x)...)

"""
    mapping(K::Type{<:Semiring}, x, y[, match::Function])

Create a sparse mapping matrix `M` such that `M[i, j] = one(K)`
if `match(x[i], y[j])` and `zero(K)` otherwise.
"""
function mapping(K::Type{<:Semiring}, x, y, match = ==)
    I, J, V = [], [], K[]
    for i in 1:length(x), j in 1:length(y)
        if match(x[i], y[j])
            push!(I, i)
            push!(J, j)
            push!(V, one(K))
        end
    end
    sparse(I, J, V, length(x), length(y))
end

#=====================================================================#

"""
    hasepsilons(fsa)

Return true if `fsa` contains epsilon nodes.
"""
hasepsilons(fsa::FSA{K, <:TransitionMatrix}) where K = false
hasepsilons(fsa::FSA{K, <:SparseLowRankMatrix}) where K = true

"""
   rmepsilon(fsa::FSA)

Remove the epsilon arcs.
"""
rmepsilon(fsa::FSA) = FSA(fsa.α, copy(fsa.T), fsa.ω, fsa.λ)

"""
    union(fsms::FSM{K}...) where K

Return the union of the given FSMs.
"""
function Base.union(fsa1::FSA{K}, fsa2::FSA{K}) where K
    FSA(
        vcat(fsa1.α, fsa2.α),
        blockdiag(fsa1.T, fsa2.T),
        vcat(fsa1.ω, fsa2.ω),
        vcat(fsa1.λ, fsa2.λ)
    )
end
Base.union(fsa1::FSA{K}, fsas::FSA{K}...) where K = foldl(union, fsas, init = fsa1)


"""
    cat(fsas::FSA...)

Return the concatenated FSAs.
"""
function Base.cat(fsa1::FSA, fsa2::FSA)
    FSA(
        vcat(fsa1.α, zero(fsa2.α)),
        blockdiag(fsa1.T, fsa2.T) + vcat(fsa1.ω, zero(fsa2.α)) * vcat(zero(fsa1.ω), fsa2.α)',
        vcat(zero(fsa1.ω), fsa2.ω),
        vcat(fsa1.λ, fsa2.λ)
    )
end
Base.cat(fsa1::FSA, fsas::FSA...)  = foldl(cat, fsas, init = fsa1)

"""
    Base.adjoint(fsa::FSA)
    fsa'

Return the reversal of `fsa`.
#"""
Base.adjoint(fsa::FSA) = FSA(fsa.ω, sparse(fsa.T'), fsa.α, fsa.λ)

"""
    renorm(fsa::FSA)

Return a normalized FSA.
"""
function renorm(::Type{Divisible}, fsa::FSA)
    Z = one(eltype(fsa.α)) ./ (sum(fsa.T, dims=2) .+ fsa.ω)
    FSA(
        fsa.α ./ sum(fsa.α),
        spdiagm(Z[:, 1]) * fsa.T,
        fsa.ω .* Z[:, 1],
        fsa.λ
    )
end
renorm(fsa::FSA{K}) where K = renorm(IsDivisible(K), fsa)

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


function Base.replace(fsa1::FSA, fsas::AbstractVector{<:FSA})
    A = blockdiag([fsaⁱ.α[:,1:1] for fsaⁱ in fsas]...)
    Ω = blockdiag([fsaⁱ.ω[:,1:1] for fsaⁱ in fsas]...)
    FSA(
        _weighted_sparse_vcat(fsa1.α, [fsaⁱ.α for fsaⁱ in fsas]),
        blockdiag([fsaⁱ.T for fsaⁱ in fsas]...) + Ω * fsa1.T * A',
        _weighted_sparse_vcat(fsa1.ω, [fsaⁱ.ω for fsaⁱ in fsas]),
        vcat([λ₁ᵢ * fsaⁱ.λ for (λ₁ᵢ, fsaⁱ) in zip(fsa1.λ, fsas)]...)
    )
end

"""
    replace(new::Function, fsa)

Replace `i`th node of `fsa` with the fsa `new(i)`.
"""
Base.replace(new::Function, fsa::FSA) =
    replace(fsa, [new(i)  for i in 1:nstates(fsa)])

"""
    propagate(fsa)

Propagate the weights along the FSA's arcs. `fsa` should be epslion
free.
"""
function propagate(fsa::FSA)
    v = fsa.α
    A = spdiagm(v) * fsa.T
    o = fsa.ω .* v
    visited = Set(findnz(v)[1])
    for n in 2:(nstates(fsa))
        v = fsa.T' * v
        A += spdiagm(v) * fsa.T
        o += fsa.ω .* v

        # Prune the states that have been visited.
        SparseArrays.fkeep!(v, (i, x) -> i ∉ visited)
        if nnz(v) > 0 push!(visited, findnz(v)[1]...) end
    end
    FSA(fsa.α, A, o, fsa.λ)
end


# Extract the non-zero states as an array of tuples.
_det_getstates(x) = [tuple(sort(map(a -> val(a)[1], collect(val(x))))...)
                     for x in nonzeros(x)]

"""
    determinize(fsa[, match])

Return an equivalent deterministic FSA. States `i` and `j` can be
merged if `match(i, j)` is `true`. Note that to guarantee the
equivalence of the returned FSA, you need to [`propagate`](@ref)
weights on `fsa` prior to call `determinize`. `fsa` should be
epsilon free.
"""
function determinize(fsa::FSA{K}, match = Base.:(==)) where K
    # We precompute the necessary matrices to estimate the new states
    # (i.e. set of states of the original fsa) and their transition weight.
    labels = sort(collect(Set(fsa.λ)), by = val)
    Mₗ = mapping(UnionConcatSemiring{LabelMonoid}, 1:nstates(fsa), labels,
                 (i, l) -> fsa.λ[i] == l)
    M = mapping(K, 1:nstates(fsa), labels, (i, l) -> fsa.λ[i] == l)
    α, T, ω = fsa.α, fsa.T, fsa.ω
    statelabels = UnionConcatSemiring{LabelMonoid}.([Set([Label(i)]) for i in 1:nstates(fsa)])
    αₗ = tobinary(UnionConcatSemiring{LabelMonoid}, fsa.α) .* statelabels
    Tₗ = tobinary(UnionConcatSemiring{LabelMonoid}, fsa.T) * spdiagm(statelabels)

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
                       nstates(fsa))
        nextstates = _det_getstates(Mₗ' * Tₗ' * zₗ)

        z = sparsevec(collect(state), one(K), nstates(fsa))
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

    # We build the actual fsa from the result of the powerset
    # construction.
    state2idx = Dict(s => i for (i, s) in enumerate(keys(newstates)))
    newlabels = [fsa.λ[s[1]] for s in keys(newstates)]
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
    FSA(α₂, T₂, ω₂, newlabels)
end

"""
    minimize(fsa)

Return an equivalent minimal FSA. Note that to guarantee the
equivalence of the returned FSA, you need to [`propagate`](@ref) weight
on `fsa` prior to call `minimize`. `fsa` should be epsilon free.
"""
minimize = adjoint ∘ determinize ∘ adjoint ∘ determinize

"""
    totalcumsum(α, T, ω, n)

Partial total cumulative sum algorithm.
"""
function totalcumsum(α, T, ω, n)
    v = α
    total = dot(v, ω)
    for i in 2:n
        v = T' * v
        total += dot(v, ω)
    end
    total
end

"""
    totalsum(α, T, ω, n)

Partial total sum algorithm.
"""
function totalsum(α, T, ω, n)
    v = α
    for i in 2:n
        v = T' * v
    end
    dot(v, ω)
end

"""
    totalweightsum(fsa::FSA, n)

Compute the `n`th partial total weight sum of `fsa`.
"""
totalweightsum(fsa::FSA, n = nstates(fsa)) = totalcumsum(fsa.α, fsa.T, fsa.ω, n)

"""
    totallabelsum(fsa::FSA, n)

Compute the `n`th partial total label sum of `fsa`.
"""
function totallabelsum(fsa::FSA, n = nstates(fsa))
    λ = UnionConcatSemiring.([Set([LabelMonoid(val(λᵢ))]) for λᵢ in fsa.λ])
    totalcumsum(
         tobinary(UnionConcatSemiring{LabelMonoid}, fsa.α) .* λ,
         tobinary(UnionConcatSemiring{LabelMonoid}, fsa.T) * spdiagm(λ),
         tobinary(UnionConcatSemiring{LabelMonoid}, fsa.ω),
         n
   )
end

