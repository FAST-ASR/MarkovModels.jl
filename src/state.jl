# MarkovModels - State of a FSM.
#
# Lucas Ondel, 2021

# A state is composed of a state id, a pdf index, a label and an
# array of link to the next states. The index and the label can be
# 'nothing'.

#######################################################################
# Types

"""
    const PdfIndex = Union{UInt64, Nothing}

Type of the state pdf index.
"""
const PdfIndex = Union{UInt64, Nothing}

"""
    const Label = Union{AbstractString, Nothing}
Type of the state label.
"""
const Label = Union{AbstractString, Nothing}

"""
    InitStateID

A type with no fields whose singleton instance [`initstateid`](@ref)
is used to represent the identifier of an initial state in a graph.
See also [`FinalStateID`](@ref).
"""
struct InitStateID end

"""
    FinalStateID

A type with no fields whose singleton instance [`finalstateid`](@ref)
is used to represent the identifier of a final state in a graph.
See also [`InitStateID`](@ref).
"""
struct FinalStateID end


"""
    const StateID = Union{Int64, InitStateID, FinalStateID}

Type of the state identifier.
"""
const StateID = Union{UInt64, InitStateID, FinalStateID}

#######################################################################
# Unique instances

"""
    initstateid

Singleton instance of type [`InitStateID`](@ref) representing the
identifier of an initial state in a graph. See also [`finalstateid`](@ref).
"""
const initstateid = InitStateID()

"""

Singleton instance of type [`FinalStateID`](@ref) representing the
identifier of a final state in a graph. See also [`initstateid`](@ref).
"""
const finalstateid = FinalStateID()

#######################################################################
# State definition

"""
    struct State
        id
        pdfindex
        label
    end

State of a FSM.
  * `id` is the unique identifier of the state within a FSM.
  * `pdfindex` is the index of a probability density associated to the
     state. If the state is non-emitting, `pdfindex` is equal to
     `nothing`.
  * `label` is a readable name (either `String` or `Nothing`).
# Examples
```julia-repl
julia> State(1)
State(1)
julia> State(1, pdfindex = 2)
State(1, pdfindex = 2)
```
"""
struct State <: AbstractState
    id::StateID
    pdfindex::PdfIndex
    label::Label
    links::Vector{<:AbstractLink}
end
State(id; pdfindex = nothing, label = nothing) = State(id, pdfindex, label,
                                                       Vector{Link}())

#######################################################################
# State interface

"""
    isemitting(state)

Returns `true` if `state` a pdf index associated.
"""
isemitting(s::AbstractState) = ! isnothing(s.pdfindex)

"""
    isinit(state)

Returns `true` if the `state` is the initial state of the FSM.
"""
isinit(s::State) = s.id == initstateid

"""
    isfinal(state)

Returns `true` if the `state` is the final state of the FSM.
"""
isfinal(s::State) = s.id == finalstateid

"""
    islabeled(state)

Returns `true` if the `state` has a label.
"""
islabeled(s::State) = ! isnothing(s.label)

