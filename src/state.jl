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
struct State{T<:SemiField} <: AbstractState
    id::StateID
    pdfindex::Union{UInt64, Nothing}
    label::Union{String, Nothing}
    links::Vector{Link{State{T},T}}
end
State{T}(id; pdfindex = nothing, label = nothing) where T<:SemiField=
    State{T}(id, pdfindex, label, Vector{Link{State{T}, T}}())

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
links(state::State) = state.links

struct EmittingStatesIterator{T<:SemiField}
    state::State{T}
end

function Base.iterate(iter::EmittingStatesIterator{T}, itstate = nothing) where T<:SemiField
    if isnothing(itstate)
        visited = Set([iter.state])
        stack = [(iter.state, one(T))]
        estates = eltype(stack)[]
        itstate  = (stack, estates, visited)
    end

    stack, estates, visited = itstate

    if ! isempty(estates) return (popfirst!(estates), (stack, estates, visited)) end
    if isempty(stack) return nothing end

    state, weight = popfirst!(stack)
    for link in links(state)
        if isemitting(link.dest)
            push!(estates, (link.dest, weight * link.weight))
        elseif link.dest ∉ visited
            push!(stack, (link.dest, weight * link.weight))
            push!(visited, link.dest)
        end
    end
    iterate(iter, (stack, estates, visited))
end

"""
    nextemittingstates(fsm, state)

Iterator over the next emitting states. For each value, the iterator
return a tuple `(nextstate, weightpath, path)`. The weight path is the
sum of the weights for all the link to reach `nextstate`. Path is a
vector of links between `state` and `nextstate`.
"""
nextemittingstates(state::AbstractState) = EmittingStatesIterator(state)

