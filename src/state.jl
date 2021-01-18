# MarkovModels - State of a FSM.
#
# Lucas Ondel, 2021

# A state is composed of a state id, a pdf index, a label and an
# array of link to the next states. The index and the label can be
# 'nothing'.

#######################################################################
# AbstractState interface

"""
    isemitting(state)

Returns `true` if `state` a pdf index associated.
"""
isemitting

"""
    isinit(state)

Returns `true` if the `state` is the initial state of the FSM.
"""
isinit

"""
    isfinal(state)

Returns `true` if the `state` is the final state of the FSM.
"""
isfinal

"""
    islabeled(state)

Returns `true` if the `state` has a label.
"""
islabeled

"""
    links(state)

Iterator over the links to the children (i.e. next states) of `state`.
"""
links

"""
    nextemittingstates(fsm, state)

Iterator over the next emitting states. For each value, the iterator
return a tuple `(nextstate, weightpath, path)`. The weight path is the
sum of the weights for all the link to reach `nextstate`. Path is a
vector of links between `state` and `nextstate`.
"""
nextemittingstates

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

Base.show(io::IO, ::InitStateID) = print(io, "initstateid")

"""
    FinalStateID

A type with no fields whose singleton instance [`finalstateid`](@ref)
is used to represent the identifier of a final state in a graph.
See also [`InitStateID`](@ref).
"""
struct FinalStateID end

Base.show(io::IO, ::FinalStateID) = print(io, "finalstateid")


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

Base.isless(UInt64, ::FinalStateID) = true
Base.isless(::FinalStateID, ::UInt64) = false
Base.isless(::UInt64, ::InitStateID) = false
Base.isless(::InitStateID, ::UInt64) = true
Base.isless(::FinalStateID, ::InitStateID) = false
Base.isless(::InitStateID, ::FinalStateID) = true

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

function Base.show(io::IO, s::State)
    print(io, "State($(s.id), $(s.pdfindex), $(s.label), Link[...])")
end

Base.:(==)(s1::State, s2::State) = s1.id == s2.id
Base.hash(s::State, h::UInt) = hash(s.id, h)

#######################################################################
# AbstractState interface implementation

isemitting(s::AbstractState) = ! isnothing(s.pdfindex)
isinit(s::State) = s.id == initstateid
isfinal(s::State) = s.id == finalstateid
islabeled(s::State) = ! isnothing(s.label)
links(state::State) = state.links

function nextemittingstates(start_state::AbstractState; return_finalstate = false)
    retval = []
    stack = [(start_state, 0.0, State[])]
    while ! isempty(stack)
        state, weight, path = popfirst!(stack)
        for link in links(state)
            if isemitting(link.dest) || (return_finalstate && isfinal(link.dest))
                push!(retval, (link.dest, weight + link.weight, [path..., state]))
            else
                push!(stack, (link.dest, weight + link.weight, [path..., state]))
            end
        end
    end
    retval
end

