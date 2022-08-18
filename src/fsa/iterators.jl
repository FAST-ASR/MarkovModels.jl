# SPDX-License-Identifier: MIT

"""
    FSAIterator(fsa)

Default FSA iterator starting from the initial states.
"""
struct FSAIterator{F}
    fsa::F
end

Base.iterate(it::FSAIterator) = (it.fsa.α, (it.fsa.T', it.fsa.α))

function Base.iterate(::FSAIterator, state)
    Tᵀ, vn_1 = state
    if num_activestates(vn_1) > 0
        vn = Tᵀ * vn_1
        return (vn, (Tᵀ, vn))
    end
end

"""
    AcyclicFSAIterator(fsa)

FSA iterator that does not follow arcs of states already visited.
"""
struct AcyclicFSAIterator{F}
    fsa::F
end

Base.iterate(it::AcyclicIterator) =
    (it.fsa.α, (Set(), it.fsa.T', it.fsa.α))

function Base.iterate(::AcyclicIterator, state)
    visited, Tᵀ, vn_1 = state

    vn_1 = prune(vn_1) do state, weight
        retval = state ∉ visited
        push!(visited, state)
        retval
    end

    vn = Tᵀ * vn_1
    return (vn, (visited, Tᵀ, vn))
end

