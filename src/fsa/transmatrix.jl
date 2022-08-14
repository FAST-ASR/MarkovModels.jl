# SPDX-License-Identifier: MIT

"""
    TransitionMatrix

Abstract type matrix representation of the arcs withing an FSA.

# Interface
- [`Base.similar`](@ref)
"""
const TransitionMatrix{K} = AbstractMatrix{K}

"""
    arcs(T::TransitionMatrix) -> I, J, V

Return a triplet `I, J, V': the list of source, destination and arc
weights respectively.
"""
arcs


"""
    reorder(T::TransitionMatrix, mapping)

Reorder the states of the FSA following `mapping[s] -> i` where `s` is
the current state number and `i` is the new state number. If a state
is omitted, it is removed from the transition matrix.
"""
reorder(::TransitionMatrix, ::AbstractDict)

#======================================================================
TransitionMatrix interface for sparse CSC matrix.
======================================================================#

arcs(T::SparseMatrixCSC) = findnz(T)

function reorder(T::SparseMatrixCSC, mapping)
    Ix, Jx, Vx = arcs(T)
    Iy, Jy, Vy = eltype(Ix)[], eltype(Jx)[], eltype(Vx)[]
    for (ix, jx, vx) in zip(Ix, Jx, Vx)
        (ix ∉ keys(mapping) || jx ∉ keys(mapping)) && continue

        push!(Iy, mapping[ix])
        push!(Jy, mapping[jx])
        push!(Vy, vx)
    end
    sparse(Iy, Jy, Vy, length(mapping), length(mapping))
end

