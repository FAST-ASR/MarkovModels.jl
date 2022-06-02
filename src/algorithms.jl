# SPDX-License-Identifier: MIT

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
    totalweightsum(fsm::FSM, n)

Compute the `n`th partial total weight sum of `fsm`.
"""
totalweightsum(fsm::FSM, n = nstates(fsm)) = totalcumsum(fsm.α, fsm.T, fsm.ω, n)

"""
    totallabelsum(fsm::FSM, n)

Compute the `n`th partial total label sum of `fsm`.
"""
function totallabelsum(fsm::FSM, n = nstates(fsm))
    λ = UnionConcatSemiring.([Set([LabelMonoid(val(λᵢ))]) for λᵢ in fsm.λ])
    totalcumsum(
         tobinary(UnionConcatSemiring{LabelMonoid}, fsm.α) .* λ,
         tobinary(UnionConcatSemiring{LabelMonoid}, fsm.T) * spdiagm(λ),
         tobinary(UnionConcatSemiring{LabelMonoid}, fsm.ω),
         n
   )
end

