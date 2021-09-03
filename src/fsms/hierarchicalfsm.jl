# SPDX-License-Identifier: MIT

function rightlabel(label)
    typeof(label) <: Tuple ?  rightlabel(label[end]) : label
end

"""
    struct HierarchicalFSM{T} <: AbstractFSM{T}
        fsm::AbstractFSM{T}
        state_mapping::Dict{Any,<:AbstractFSM{T}}
    end

FSM where each state is an FSM. This FSM is used to "compose" different
FSMs, e.g. a lexicon and a language model.
"""
struct HierarchicalFSM{T} <: AbstractFSM{T}
    fsm::AbstractFSM{T}
    state_mapping::Dict
end

function makestate(s1, s2)
    label = if typeof(s1.label) <: Tuple && typeof(s2.label) <: Tuple
        (s1.label..., s2.label...)
    elseif typeof(s1.label) <: Tuple
        (s1.label..., s2.label)
    elseif typeof(s2.label) <: Tuple
        (s1.label, s2.label...)
    else
        (s1.label, s2.label)
    end
    State((s1..., s2...), label,
          s1.initweight * s2.initweight, s1.finalweight * s2.finalweight)
end

function states(hfsm::HierarchicalFSM{T}) where T
    retval = []
    for s1 in states(hfsm.fsm)
        for s2 in states(hfsm.state_mapping[rightlabel(s1.label)])
            push!(retval, makestate(s1, s2))
        end
    end
    retval
end

function arcs(hfsm::HierarchicalFSM{T}, state::State{T}) where T
    s1, s2 = state.id

    retval = []
    for arc in arcs(hfsm.state_mapping[rightlabel(s1.label)], s2)
        push!(retval, Arc(makestate(s1, arc.dest), arc.weight))
    end

    if isfinal(s2)
        for arc in arcs(hfsm.fsm, s1)
            mlabel = rightlabel(arc.dest.label)
            for s in filter(isinit, states(hfsm.state_mapping[mlabel]))
                weight = s2.finalweight * arc.weight * s.initweight
                push!(retval, Arc(makestate(arc.dest, s), weight))
            end
        end
    end
    retval
end

