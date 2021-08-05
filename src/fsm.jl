# SPDX-License-Identifier: MIT

const PdfIndex = Union{Int,Nothing}
const Label = Union{AbstractString,Nothing}

mutable struct State{T<:Semifield}
    id::Int
    initweight::T
    finalweight::T
    pdfindex::PdfIndex
    label::Label
end

isinit(s::State{T}) where T = s.initweight ≠ zero(T)
isfinal(s::State{T}) where T = s.finalweight ≠ zero(T)
islabeled(s::State) = ! isnothing(s.label)
isemitting(s::State)  = ! isnothing(s.pdfindex)
setinit!(s::State{T}, weight::T = one(T)) where T = s.initweight = weight
setfinal!(s::State{T}, weight::T = one(T)) where T = s.finalweight = weight

mutable struct Link{T<:Semifield}
    dest::State
    weight::T
end

"""
    struct FSM{T<:Semifield}
        states # vector of states
        links # Dict state -> vector of links
    end

Probabilistic finite state machine.
"""
struct FSM{T<:Semifield}
    states::Vector{State{T}}
    links::Dict{State, Vector{Link{T}}}
end
FSM{T}() where T = FSM{T}(State{T}[], Dict{State, Vector{Link{T}}}())
FSM() = FSM{LogSemifield{Float64}}()

states(fsm::FSM) = fsm.states
links(fsm::FSM{T}, state::State{T}) where T = get(fsm.links, state, Link{T}[])

function addstate!(fsm::FSM{T}; initweight = zero(T), finalweight = zero(T),
                   pdfindex = nothing, label = nothing) where T
    s = State(length(fsm.states)+1, initweight, finalweight, pdfindex, label)
    push!(fsm.states, s)
    s
end

function link!(fsm::FSM{T}, src::State{T}, dest::State{T}, weight::T = one(T)) where T
    list = get(fsm.links, src, Link{T}[])
    link = Link{T}(dest, weight)
    push!(list, link)
    fsm.links[src] = list
    link
end

function Base.show(io, ::MIME"image/svg+xml", fsm::FSM)
    dotpath, dotfile = mktemp()
    svgpath, svgfile = mktemp()

    write(dotfile, "Digraph {\n")
    write(dotfile, "rankdir=LR;")

    for s in states(fsm)
        name = "$(s.id)"
        label = islabeled(s) ? "$(s.label)" : "ϵ"
        label *= isemitting(s) ? ":$(s.pdfindex)" : ":ϵ"
        if s.initweight ≠ zero(typeof(s.initweight))
            weight = round(convert(Float64, s.initweight), digits = 3)
            label *= "/$(weight)"
        end
        if s.finalweight ≠ zero(typeof(s.finalweight))
            weight = round(convert(Float64, s.finalweight), digits = 3)
            label *= "/$(weight)"
        end
        attrs = "shape=" * (isfinal(s) ? "doublecircle" : "circle")
        attrs *= " penwidth=" * (isinit(s) ? "2" : "1")
        attrs *= " label=\"" * label * "\""
        attrs *= " style=filled fillcolor=" * (isemitting(s) ? "lightblue" : "none")
        write(dotfile, "$name [ $attrs ];\n")
    end

    for src in states(fsm)
        for link in links(fsm, src)
            weight = round(convert(Float64, link.weight), digits = 3)
            srcname = "$(src.id)"
            destname = "$(link.dest.id)"
            write(dotfile, "$srcname -> $destname [ label=\"$(weight)\" ];\n")
        end
    end
    write(dotfile, "}\n")
    close(dotfile)
    run(`dot -Tsvg $(dotpath) -o $(svgpath)`)

    xml = read(svgfile, String)
    write(io, xml)

    close(svgfile)

    rm(dotpath)
    rm(svgpath)
end

#======================================================================
FSM operations
======================================================================#

struct InvalidFSMError <: Exception
    msg
end
Base.show(io::IO, e::InvalidFSMError) = print(io, "invalid input FSM: $(e.msg)")

"""
    union(fsm1, fsm2, ...)

Merge all the fsms into a single one.
"""
function Base.union(fsm1::FSM{T}, fsm2::FSM{T}) where T
    allstates = union(states(fsm1), states(fsm2))
    newfsm = FSM{T}()

    smap = Dict()
    for state in allstates
        smap[state] = addstate!(newfsm, label = state.label,
                                pdfindex = state.pdfindex,
                                initweight = state.initweight,
                                finalweight = state.finalweight)
    end

    for src in states(fsm1)
        for link in links(fsm1, src)
            link!(newfsm, smap[src], smap[link.dest], link.weight)
        end
    end

    for src in states(fsm2)
        for link in links(fsm2, src)
            link!(newfsm, smap[src], smap[link.dest], link.weight)
        end
    end

    newfsm
end
Base.union(f::FSM{T}, o::FSM{T}...) where T = foldl(union, o, init = f)

"""
    renormalize!(fsm)

Ensure the that all the weights of all the outgoing arcs leaving a
state sum up to 1.
"""
function renormalize!(fsm::FSM{T}) where T
    total = zero(T)
    for s in filter(isinit, states(fsm)) total += s.initweight end
    for s in filter(isinit, states(fsm)) s.initweight /= total end

    for src in states(fsm)
        total = src.finalweight
        for link in links(fsm, src) total += link.weight end
        for link in links(fsm, src) link.weight /= total end
        src.finalweight /= total
    end

    fsm
end

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
                for link in links(subfsms[matchlabel(s.label)], cs)
                    link!(newfsm, smap[cs], smap[link.dest], link.weight)
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
        for link in links(fsm, osrc)
            src = smap_out[osrc]
            dest = smap_in[link.dest]
            link!(newfsm, src, dest, link.weight * weight)
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
    newlinks = Dict()
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
            for link in links(fsm, ls)
                push!(nextstates, (link.dest, link.weight))
            end
        end

        nextlabels = _unique_labels(nextstates, T, step+1, init = false)
        for (key2, value2) in nextlabels
            w = get(newlinks, (key,key2), zero(T))
            newlinks[(key,key2)] = w+value2[end]
            if key2 ∉ visited
                queue[key2] = value2
                push!(visited, key2)
            end
        end
    end

    for (key, value) in newlinks
        src = smap[key[1]]
        dest = smap[key[2]]
        weight = value
        link!(newfsm, src, dest, weight)
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
        for link in links(fsm, src)
            link!(newfsm, smap[link.dest], smap[src], link.weight)
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
) where T <: Semifield

    if state in visited
        return closure
    end
	push!(visited, state)

    for l in links(fsm, state)
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
function remove_eps(fsm::FSM{T}) where T <: Semifield
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
        for l in links(fsm, s)
            if isemitting(s) && isemitting(l.dest)
                link!(nfsm, smap[s], smap[l.dest], l.weight)
            elseif isemitting(s)
                for (ns, w) in eps_closures[l.dest]
                    link!(nfsm, smap[s], smap[ns], l.weight * w)
                end
            end
        end
    end
    return nfsm
end

