# MarkovModels.jl
#
# Lucas Ondel, 2021
#

include("state.jl")
include("link.jl")

mutable struct StateIDCounter
    count::UInt64
end

struct FSM{T}
    idcounter::StateIDCounter
    states::Set{State}
    links::Dict{State, Vector{Link{T}}}
end
FSM{T}() where T = FSM{T}(StateIDCounter(0), Set{State}(), Dict{State, Vector{Link{T}}}())

function addstate!(fsm::FSM{T}) where T
    fsm.idcounter.count += 1
    s = State(fsm.idcounter.count, zero(T), zero(T))
    push!(fsm.states, s)
    s
end

"""
    link!(fsm::FSM{T}, src, dest[, weight = zero(T)])

Add a weighted connection between `state1` and `state2`.
"""
function link!(fsm::FSM{T}, src::State{T}, dest::State{T}; ilabel::Label = nothing,
               olabel::Label = nothing, weight::T = one(T)) where T
    list = get(fsm.links, src, Link{T}[])
    push!(list, Link{T}(dest, ilabel, olabel, weight))
    fsm.links[src] = list
end

initstates(fsm::FSM{T}) where T = filter(s -> s.startweight ≠ zero(T), fsm.states)
finalstates(fsm::FSM{T}) where T = filter(s -> s.endweight ≠ zero(T), fsm.states)

"""
    finalstate(fsm)

Returns the final state of `fsm`.
"""
#finalstate(fsm::FSM) = shared_finalstate

"""
    states(fsm)

Iterator over the state of `fsm`.
"""
states(fsm::FSM) = fsm.states

"""
    links(fsm, state)

Iterator over the links to the children (i.e. next states) of `state`.
"""
links(fsm::FSM{T}, state::State) where T = get(fsm.links, state, Link{T}[])

