# SPDX-License-Identifier: MIT

"""
    struct HierarchicalFSM{T} <: AbstractFSM{T}
        fsm::AbstractFSM{T}
        state_mapping::Dict{Any,<:AbstractFSM{T}}
    end

FSM where each state is an FSM.
"""
struct HierarchicalFSM{T} <: AbstractFSM{T}
    fsm::AbstractFSM{T}
    state_mapping::Dict{<:Any,<:AbstractFSM{T}}
end

function makestate(s1, s2)
    State((s1, s2), (s1.label, s2.label), s1.initweight * s2.initweight,
          s1.finalweight * s2.finalweight)
end

function matchlabel(label)
    typeof(label) <: Tuple ?  label[end] : label
end

function states(hfsm::HierarchicalFSM{T}) where T
    retval = []
    for s1 in states(hfsm.fsm)
        for s2 in states(hfsm.state_mapping[matchlabel(s1.label)])
            push!(retval, makestate(s1, s2))
        end
    end
    retval
end

function arcs(hfsm::HierarchicalFSM{T}, state::State{T}) where T
    s1, s2 = state.id

    retval = []
    for arc in arcs(hfsm.state_mapping[matchlabel(s1.label)], s2)
        push!(retval, Arc(makestate(s1, arc.dest), arc.weight))
    end

    if isfinal(s2)
        for arc in arcs(hfsm.fsm, s1)
            mlabel = matchlabel(arc.dest.label)
            for s in filter(isinit, states(hfsm.state_mapping[mlabel]))
                weight = s2.finalweight * arc.weight * s.initweight
                push!(retval, Arc(makestate(arc.dest, s), weight))
            end
        end
    end
    retval
end

