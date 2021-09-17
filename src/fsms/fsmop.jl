# SPDX-License-Identifier: MIT

"""
    Base.union(f::AbstractFMS{T}...)

Take the union of the provided FSMs.
"""
function Base.union(fsm1::AbstractFSM{T}, fsm2::AbstractFSM{T}) where T
    allstates = union(collect(states(fsm1)), collect(states(fsm2)))
    newfsm = VectorFSM{T}()

    smap1 = Dict()
    for state in states(fsm1)
        smap1[state] = addstate!(newfsm, state.label,
                                initweight = state.initweight,
                                finalweight = state.finalweight)
    end

    smap2 = Dict()
    for state in states(fsm2)
        smap2[state] = addstate!(newfsm, state.label,
                                initweight = state.initweight,
                                finalweight = state.finalweight)
    end

    for src in states(fsm1)
        for arc in arcs(fsm1, src)
            addarc!(newfsm, smap1[src], smap1[arc.dest], arc.weight)
        end
    end

    for src in states(fsm2)
        for arc in arcs(fsm2, src)
            addarc!(newfsm, smap2[src], smap2[arc.dest], arc.weight)
        end
    end

    newfsm
end
Base.union(f::AbstractFSM{T}, o::AbstractFSM{T}...) where T =
    foldl(union, o, init = f)

"""
    renormalize(fsm::AbstractFSM{T}) where T<:Semifield

Ensure the that the weights of all the outgoing arcs leaving a
state sum up to `one(T)`.
"""
function renormalize(fsm::AbstractFSM{T}) where T<:Semifield
    total_initweight = zero(T)
    for s in filter(isinit, states(fsm)) total_initweight += s.initweight end

    totals = Dict()
    for s in states(fsm)
        v = s.finalweight
        for a in arcs(fsm, s)
            v += a.weight
        end
        totals[s] = v
    end

    newfsm = VectorFSM{T}()
    smap = Dict()
    for s in states(fsm)
        iw = total_initweight == zero(T) ? zero(T) : s.initweight / total_initweight
        fw = totals[s] == zero(T) ? zero(T) : s.finalweight / totals[s]
        smap[s] = addstate!(newfsm, s.label, initweight = iw, finalweight = fw)
    end

    for s in states(fsm)
        for a in arcs(fsm, s)
            addarc!(newfsm, smap[s], smap[a.dest], a.weight / totals[s])
        end
    end

    newfsm
end

function next_states(fsm, set_states)
    nextstates = []
    for s in set_states
        for arc in arcs(fsm, s)
            push!(nextstates, (arc.dest, arc.weight))
        end
    end
    nextstates
end

function nextlabels(T, statelist; init = false)
    labels = Dict()
    for (s, w) in statelist
        set_states, iw, fw, sw = get(labels, s.label, (Set(), zero(T), zero(T), zero(T)))
        push!(set_states, s)
        labels[s.label] = (set_states, iw+s.initweight, fw+s.finalweight, sw+w)
    end

    retval = []
    for (label, (set_states, iw, fw, sw)) in labels
        if ! init
            push!(retval, (label, set_states, zero(T), fw, sw))
        else
            push!(retval, (label, set_states, iw, fw, sw))
        end
    end
    retval
end

"""
    determinize(fsm::AbstractFSM)

Determinize the FSM w.r.t. the state labels. A FSM is deterministic
if for any state there is no next states with the same label.
"""
function determinize(fsm::AbstractFSM{T}; renormalize_output = true) where T
    dfsm = VectorFSM{T}()
    smap = Dict()
    newarcs = Dict()
    visited = Set()

    init_ws = [(s, one(T)) for s in filter(isinit, collect(states(fsm)))]
    queue = nextlabels(T, init_ws; init = true)
    while ! isempty(queue)
        label, set_states, iw, fw, w = popfirst!(queue)

        nl= nextlabels(T, next_states(fsm, set_states); init = false)
        for (next_label, next_set_states, next_iw, next_fw, next_w) in nl
            newarcs[(set_states, next_set_states)] = next_w
            if next_set_states ∉ visited
                val = (next_label, next_set_states, next_iw, next_fw, next_w)
                push!(queue, val)
                push!(visited, next_set_states)
            end
        end

        if set_states ∉ keys(smap)
            smap[set_states] = addstate!(dfsm, label, initweight = iw, finalweight = fw)
        end
    end

    for ((src, dest), weight) in newarcs
        addarc!(dfsm, smap[src], smap[dest], weight)
    end

    renormalize_output ? dfsm |> renormalize : dfsm
end
unnorm_determinize(fsm) = determinize(fsm, renormalize_output = false)

function previous_states(r_arc_map, set_states)
    nextstates = []
    for s in set_states
        for (src, dest, weight) in r_arc_map[s]
            push!(nextstates, (src, weight))
        end
    end
    nextstates
end

function reverse_determinize(fsm::AbstractFSM{T}) where T
    r_arc_map = Dict()
    for s in states(fsm)
        for a in arcs(fsm, s)
            arclist = get(r_arc_map, a.dest, [])
            push!(arclist, (s, a.dest, a.weight))
            r_arc_map[a.dest] = arclist
        end
    end

    dfsm = VectorFSM{T}()
    smap = Dict()
    newarcs = Dict()
    visited = Set()

    init_ws = [(s, one(T)) for s in filter(isfinal, collect(states(fsm)))]
    queue, _ = nextlabels(T, init_ws; init = false)
    while ! isempty(queue)
        label, set_states, iw, fw, w = popfirst!(queue)

        nl, total = nextlabels(T, previous_states(r_arc_map, set_states);
                               init = true, finalweight = fw)
        for (next_label, next_set_states, next_iw, next_fw, next_w) in nl
            newarcs[(set_states, next_set_states)] = next_w
            if next_set_states ∉ visited
                val = (next_label, next_set_states, next_iw, next_fw, next_w)
                push!(queue, val)
                push!(visited, next_set_states)
            end
        end

        if set_states ∉ keys(smap)
            #nfw = (total ≠ zero(T) ) ? fw / total : fw
            nfw = fw
            smap[set_states] = addstate!(dfsm, label, initweight = iw,
                                         finalweight = nfw)
        end
    end

    for ((dest, src), weight) in newarcs
        addarc!(dfsm, smap[src], smap[dest], weight)
    end

    dfsm
end

"""
    transpose(fsm::AbstractFSM)

Reverse the direction of the arcs and, for each state, inverse the
initial and final weight.
"""
function Base.transpose(fsm::AbstractFSM{T}) where T
    newfsm = VectorFSM{T}()
    smap = Dict()
    for s in states(fsm)
        ns = addstate!(newfsm, s.label, initweight = s.finalweight,
                       finalweight = s.initweight)
        smap[s] = ns
    end

    for src in states(fsm)
        for arc in arcs(fsm, src)
            addarc!(newfsm, smap[arc.dest], smap[src], arc.weight)
        end
    end

    newfsm
end

"""
    minimize(fsm::AbstractFSM)

Return a minimal equivalent fsm.
"""
minimize(fsm::AbstractFSM{T}) where T =
    (renormalize ∘ transpose ∘ unnorm_determinize ∘
     transpose ∘ unnorm_determinize)(fsm)

"""
    label_closure!(closure, fsm, state, label;  [weight=1], [visited=[]])

Find label closure from `state` in `fsm`.
"""
function label_closure!(
        closure::Vector, fsm::AbstractFSM{T}, state::State, label;
        weight::T=one(T), visited::Vector{State} = State[]
) where T <: Semifield

    if state in visited
        return closure
    end
	push!(visited, state)

    for l in arcs(fsm, state)
        if l.dest.label != label
            push!(closure, (l.dest, l.weight * weight))
        else
            label_closure!(closure, fsm, l.dest, label; weight=l.weight * weight, visited=visited)
        end
    end
    return closure
end

"""
	remove_label(fsm, label)

Removes all states from`fsm` with label `label`.
"""
function remove_label(fsm::AbstractFSM{T}, label) where T <: Semifield
    nfsm = VectorFSM{T}()
    label_states = []

    iw, fw = Dict(), Dict()
    for s in states(fsm)
        if s.label != label
            iw[s] = s.initweight
            fw[s] = s.finalweight
        end
    end

    label_closures = Dict{State, Vector}()
    for s in states(fsm)
        if s.label == label
            closure = label_closure!([], fsm, s)
            label_closures[s] = unique!(closure)

            for ns in label_closures[s]
                iw[ns] = iw[ns] + s.initweight
                fw[ns] = fw[ns] + s.finalweight
            end
        end
    end

    smap = Dict{State, State}()
    for s in keys(iw)
        addstate!(nfsm, s.label; initweight = iw[s], finalweight = fw[s])
    end

    for s in states(fsm)
        for l in arcs(fsm, s)
            if s.label != label && l.dest.label != label
                addarc!(nfsm, smap[s], smap[l.dest], l.weight)
            elseif s.label != label
                for (ns, w) in label_closures[l.dest]
                    addarc!(nfsm, smap[s], smap[ns], l.weight * w)
                end
            end
        end
    end
    return nfsm
end
