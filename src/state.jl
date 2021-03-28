# MarkovModels.jl
#
# Lucas Ondel, 2021

struct InitStateID end
const initstateid = InitStateID()
Base.show(io::IO, ::InitStateID) = print(io, "initstateid")

struct FinalStateID end
const finalstateid = FinalStateID()
Base.show(io::IO, ::FinalStateID) = print(io, "finalstateid")

const StateID = Union{UInt64, InitStateID, FinalStateID}

# The initstateid (finalstateid) is lower (greater) than all other ids.
Base.isless(UInt64, ::FinalStateID) = true
Base.isless(::FinalStateID, ::UInt64) = false
Base.isless(::UInt64, ::InitStateID) = false
Base.isless(::InitStateID, ::UInt64) = true
Base.isless(::FinalStateID, ::InitStateID) = false
Base.isless(::InitStateID, ::FinalStateID) = true

"""
    struct State
        id
        pdfindex
    end

State of a FSM.
  * `id` is the unique identifier of the state within a FSM.
  * `pdfindex` is the index of a probability density associated to the
     state. If the state is non-emitting, `pdfindex` is equal to
     `nothing`.

# Examples
```julia-repl
julia> State(1)
State(1)
julia> State(1, pdfindex = 2)
State(1, pdfindex = 2)
```
"""
struct State
    id::StateID
    pdfindex::Union{UInt64, Nothing}
end

Base.:(==)(s1::State, s2::State) = s1.id == s2.id
Base.hash(s::State, h::UInt) = hash(s.id, h)

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

"""
    links(state)

Iterator over the links to the children (i.e. next states) of `state`.
"""
#links(state::State) = state.links

