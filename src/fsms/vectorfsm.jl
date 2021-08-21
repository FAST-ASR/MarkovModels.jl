# SPDX-License-Identifier: MIT

"""
    struct VectorFSM{T} <: AbstractMutableFSM{T}
        states::Vector{State{T}}
        arcs::Dict{State, Vector{Arc{T}}}
    end

Mutable FSM that store the states and arcs into vectors.
"""
struct VectorFSM{T} <: AbstractMutableFSM{T}
    states::Vector{State{T}}
    arcs::Dict{State, Vector{Arc{T}}}
end

VectorFSM{T}() where T = VectorFSM{T}([], Dict())

states(fsm::VectorFSM) = fsm.states
arcs(fsm::VectorFSM{T}, state) where T = get(fsm.arcs, state, Arc{T}[])

function addstate!(fsm::VectorFSM{T}, label; initweight = zero(T),
                   finalweight = zero(T)) where T
    s = State(length(fsm.states)+1, label, initweight, finalweight)
    push!(fsm.states, s)
    s
end

function addarc!(fsm::VectorFSM{T}, src, dest, weight::T = one(T)) where T
    arclist = get(fsm.arcs, src, Arc{T}[])
    arc = Arc{T}(dest, weight)
    push!(arclist, arc)
    fsm.arcs[src] = arclist
    arc
end
