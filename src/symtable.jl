# SPDX-License-Identifier: MIT

"""
    SymbolTable

Abstract type vector representing the symbol for each state.

# Interface
- [`filterstates`](@ref)
"""
const SymbolTable = AbstractVector


"""
    reorder(symtable, mapping)

Reorder the `symtable` following `mapping[s] -> i` where `s` is the
current state number and `i` is the new state number. If a state is
omitted, it is removed from the symbol table. The order is assumed
to be contiguous from 1 to `length(mapping)`.
"""
reorder(symtable::AbstractVector, mapping) =
    [symtable[s] for s in sort(collect(keys(mapping)))]


#======================================================================
Default symbol table
======================================================================#

"""
    struct DefaultSymbolTable
        ...
    end

Default symbol table that return the index of the state as a label.
"""
struct DefaultSymbolTable{T} <: AbstractVector{T}
    size::Int64
end

DefaultSymbolTable(size) = DefaultSymbolTable{Int64}(size)
Base.size(st::DefaultSymbolTable) = (st.size,)
Base.IndexStyle(::Type{<:DefaultSymbolTable}) = IndexLinear()
Base.getindex(::DefaultSymbolTable{T}, i::Int) where T = T(i)

reorder(symtable::DefaultSymbolTable, mapping) =
    DefaultSymbolTable(length(mapping))


