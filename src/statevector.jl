# SPDX-License-Identifier: MIT

"""
    StateVector

Abstract type vector representing the the active states of while
iterating over an FSA. The dimension of the vector is the number
of states in the FSA and the active states have non-zero values at
their corresponding dimension.

# Interface
- [`activestates`](@ref)
- [`filterstates`](@ref)
- [`num_activestates`](@ref)
- [`prune`](@ref)
"""
const StateVector{K} = Union{Vector{K}, SparseVector{K}}


"""
    activestates(fn::Function, s::StateVector) -> I, V

Return the  active states and their associated weights.
"""
activestates


"""
    reorder(x::StateVector, mapping)

Reorder the dimension of the state vector following `mapping[s] -> i`
where `s` is the current state number and `i` is the new state number.
If a dimension is omitted, it is removed from the state vector.
"""
reorder(::StateVector, ::AbstractDict)


"""
    num_activestates(fn::Function, s::StateVector)

Return the number of active states, i.e. the number of non-zero
dimesion.
"""
num_activestates


"""
    prune(fn::Function, s::StateVector)

Return a vector similar to `s` with `i`th state active if
`fn(i, s[i]) == true`.
"""
prune(fn::Function, x::StateVector) = Base.filter(fn, x)


#======================================================================
StateVector interface for sparse vectors.
======================================================================#

activestates(x::SparseVector) = findnz(x)

function reorder(x::SparseVector, mapping)
    Ix, Vx = findnz(x)
    Iy, Vy = eltype(Ix)[], eltype(Vx)[]
    for (ix, vx) in zip(Ix, Vx)
        ix ∉ keys(mapping) && continue
        push!(Iy, mapping[ix])
        push!(Vy, vx)
    end
    sparsevec(Iy, Vy, length(mapping))
end

num_activestates(x::SparseVector) = nnz(x)

function prune(fn::Function, x::SparseVector)
    Ix, Vx = findnz(x)
    Iy, Vy = eltype(Ix)[], eltype(Vx)[]
    for (ix, vx) in zip(Ix, Vx)
        if fn(ix, vx)
            push!(Iy, ix)
            push!(Vy, vx)
        end
    end
    sparsevec(Iy, Vy, length(x))
end

