# SPDX-License-Identifier: MIT

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
    renormalize!(fsm::AbstractFSM{T})

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

#=

"""
    replace(fsm, subfsms, delim = "!")

Replace the state in `fsm` wiht a sub-fsm from `subfsms`. The pairing
is done with the last token of `label` of the state, i.e. the state
with label `a!b!c` will be replaced by `subfsms[c]`. States that don't
have matching labels are left untouched.
"""
function Base.replace(fsm::FSM{T}, subfsms::Dict, delim = "!") where T
    newfsm = FSM{T}()

    matchlabel = label -> isnothing(label) ? nothing : split(label, delim)[end]

    smap_in = Dict()
    smap_out = Dict()
    for s in states(fsm)
        if matchlabel(s.label) in keys(subfsms)
            smap = Dict()
            for cs in states(subfsms[matchlabel(s.label)])
                label = "$(s.label)$(delim)$(cs.label)"
                ns = addstate!(newfsm, pdfindex = cs.pdfindex, label = label,
                               initweight = s.initweight * cs.initweight,
                               finalweight = s.finalweight * cs.finalweight)
                smap[cs] = ns

                if isinit(cs) smap_in[s] = ns end
                if isfinal(cs) smap_out[s] = ns end
            end

            for cs in states(subfsms[matchlabel(s.label)])
                for arc in arcs(subfsms[matchlabel(s.label)], cs)
                    addarc!(newfsm, smap[cs], smap[arc.dest], arc.weight)
                end
            end

        else
            ns = addstate!(newfsm, pdfindex = s.pdfindex, label = s.label,
                           initweight = s.initweight, finalweight = s.finalweight)
            smap_in[s] = ns
            smap_out[s] = ns
        end
    end

    for osrc in states(fsm)
        weight = one(T)
        if matchlabel(osrc.label) in keys(subfsms)
            finals = filter(isfinal, states(subfsms[matchlabel(osrc.label)]))
            weight = sum(map(s->s.finalweight, finals))
        end
        for arc in arcs(fsm, osrc)
            src = smap_out[osrc]
            dest = smap_in[arc.dest]
            addarc!(newfsm, src, dest, arc.weight * weight)
        end
    end

    newfsm
end

function _unique_labels(statelist, T, step; init = true)
    labels = Dict()
    for (s, w) in statelist
        lstates, iw, fw, tw = get(labels, (s.label, step), (Set(), zero(T), zero(T), zero(T)))
        push!(lstates, s)
        labels[(s.label, step)] = (lstates, iw+s.initweight, fw+s.finalweight, tw+w)
    end

    # Inverse the map so that the set of states is the key.
    retval = Dict()
    for (key, value) in labels
        retval[value[1]] = (key[1], value[2], value[3], value[4], init)
    end
    retval
end

"""
    determinize(fsm)

Determinize the FSM w.r.t. the state labels. An error will be raised
if the fsm has emitting states.
"""
function determinize(fsm::FSM{T}) where T
    newfsm = FSM{T}()
    smap = Dict()
    newarcs = Dict()
    visited = Set()

    if length(collect(filter(isemitting, states(fsm)))) > 0
        throw(InvalidFSMError("cannot determinize FSM with emitting states."))
    end

    initstates = [(s, zero(T)) for s in filter(isinit, collect(states(fsm)))]
    queue = _unique_labels(initstates, T, 0, init = true)
    for key in keys(queue) push!(visited, key) end
    while ! isempty(queue)
        key, value = pop!(queue)
        lstates = key
        label, iw, fw, tw, init = value
        step = 0

        if key ∉ keys(smap)
            if init
                s = addstate!(newfsm, label = label, initweight = iw, finalweight = fw)
            else
                s = addstate!(newfsm, label = label, finalweight = fw)
            end
            smap[key] = s
        end

        nextstates = []
        for ls in lstates
            for arc in arcs(fsm, ls)
                push!(nextstates, (arc.dest, arc.weight))
            end
        end

        nextlabels = _unique_labels(nextstates, T, step+1, init = false)
        for (key2, value2) in nextlabels
            w = get(newarcs, (key,key2), zero(T))
            newarcs[(key,key2)] = w+value2[end]
            if key2 ∉ visited
                queue[key2] = value2
                push!(visited, key2)
            end
        end
    end

    for (key, value) in newarcs
        src = smap[key[1]]
        dest = smap[key[2]]
        weight = value
        addarc!(newfsm, src, dest, weight)
    end

    newfsm
end

"""
    transpose(fsm)

Reverse the direction of the arcs.
"""
function Base.transpose(fsm::FSM{T}) where T
    newfsm = FSM{T}()
    smap = Dict()
    for s in states(fsm)
        ns = addstate!(newfsm, label = s.label, initweight = s.finalweight,
                       finalweight = s.initweight, pdfindex = s.pdfindex)
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
    minimize(fsm)

Return a minimal equivalent fsm.
"""
minimize(fsm::FSM{T}) where T = (transpose ∘ determinize ∘ transpose ∘ determinize)(fsm)

"""
	eps_closure!(fsm, state, closure; [weight=1], [visited=[]])

Find eps closure from `state` in `fsm`.
"""
function eps_closure!(
        fsm::FSM{T}, state::State, closure::Vector;
        weight::T=one(T), visited::Vector{State} = State[]
) where T <: Semiring

    if state in visited
        return closure
    end
	push!(visited, state)

    for l in arcs(fsm, state)
        if isemitting(l.dest)
            push!(closure, (l.dest, l.weight * weight))
        else
            eps_closure!(fsm, l.dest, closure; weight=l.weight * weight, visited=visited)
        end
    end
    return closure
end

"""
	remove_eps(fsm)

Removes non-emitting states from `fsm`. An error will be raised if
a non-emitting states has a label and/or it is an initial or final
state.
"""
function remove_eps(fsm::FSM{T}) where T <: Semiring
    nfsm = FSM{T}()
    smap = Dict{State, State}()
    eps_states = []
    eps_closures = Dict{State, Vector}()
    for s in states(fsm)
        if isemitting(s)
            smap[s] = addstate!(nfsm;
                initweight=s.initweight, finalweight=s.finalweight,
                pdfindex=s.pdfindex, label=s.label)
        else
            # Some checks to make sure the resulting FSM will be
            # equivalent to the input one.
            if islabeled(s)
                throw(InvalidFSMError("cannot remove labeled non-emitting state"))
            end
            if isinit(s)
                throw(InvalidFSMError("cannot remove starting non-emitting state"))
            end
            if isfinal(s)
                throw(InvalidFSMError("cannot remove final non-emitting state"))
            end

            push!(eps_states, s)
        end
    end

    for eps in eps_states
        closure = eps_closure!(fsm, eps, [])
        eps_closures[eps] = unique!(closure)
    end

    for s in states(fsm)
        for l in arcs(fsm, s)
            if isemitting(s) && isemitting(l.dest)
                addarc!(nfsm, smap[s], smap[l.dest], l.weight)
            elseif isemitting(s)
                for (ns, w) in eps_closures[l.dest]
                    addarc!(nfsm, smap[s], smap[ns], l.weight * w)
                end
            end
        end
    end
    return nfsm
end

=#
