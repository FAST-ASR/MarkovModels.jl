# Copyright - 2020 - Brno University of Technology
# Copyright - 2021 - CNRS
#
# Contact: Lucas Ondel <lucas.ondel@gmail.com>
#
# This software is governed by the CeCILL  license under French law and
# abiding by the rules of distribution of free software.  You can  use,
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
#
# As a counterpart to the access to the source code and  rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author,  the holder of the
# economic rights,  and the successive licensors  have only  limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading,  using,  modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean  that it is complicated to manipulate,  and  that  also
# therefore means  that it is reserved for developers  and  experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and,  more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.

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

#######################################################################
# FSM operations

function renormalize!(fsm::FSM{T}) where T
    total = zero(T)
    for s in filter(isinit, states(fsm)) total += s.initweight end
    for s in filter(isinit, states(fsm)) s.initweight /= total end

    total = zero(T)
    for s in filter(isfinal, states(fsm)) total += s.finalweight end
    for s in filter(isfinal, states(fsm)) s.finalweight /= total end

    for src in states(fsm)
        total = zero(T)
        for link in links(fsm, src) total += link.weight end
        for link in links(fsm, src) link.weight /= total end
    end

    fsm
end

function Base.replace(fsm::FSM{T}, subfsms::Dict) where T
    newfsm = FSM{T}()

    smap_in = Dict()
    smap_out = Dict()
    for s in states(fsm)
        if s.label in keys(subfsms)
            smap = Dict()
            for cs in states(subfsms[s.label])
                label = "$(s.label)!$(cs.label)"
                ns = addstate!(newfsm, pdfindex = cs.pdfindex, label = label,
                               initweight = s.initweight * cs.initweight,
                               finalweight = s.finalweight * cs.finalweight)
                smap[cs] = ns

                if isinit(cs) smap_in[s] = ns end
                if isfinal(cs) smap_out[s] = ns end
            end

            for cs in states(subfsms[s.label])
                for link in links(subfsms[s.label], cs)
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
        for link in links(fsm, osrc)
            src = smap_out[osrc]
            dest = smap_in[link.dest]
            link!(newfsm, src, dest, link.weight)
        end
    end

    newfsm
end

function _unique_labels(statelist, T, step)
    labels = Dict()
    for (s, w) in statelist
        lstates, iw, fw, tw = get(labels, (s.label, step), (Set(), zero(T), zero(T), zero(T)))
        push!(lstates, s)
        labels[(s.label, step)] = (lstates, iw+s.initweight, fw+s.finalweight, tw+w)
    end
    labels
end

function determinize(fsm::FSM{T}) where T
    newfsm = FSM{T}()
    smap = Dict()
    newlinks = Dict()

    initstates = [(s, zero(T)) for s in filter(isinit, collect(states(fsm)))]
    queue = _unique_labels(initstates, T, 0)
    while ! isempty(queue)
        key, value = pop!(queue)
        label, step = key
        lstates, iw, fw, tw = value

        s = addstate!(newfsm, label = label, initweight = iw, finalweight = fw)
        smap[key] = s

        nextstates = []
        for ls in lstates
            for link in links(fsm, ls)
                push!(nextstates, (link.dest, link.weight))
            end
        end

        nextlabels = _unique_labels(nextstates, T, step+1)
        for (key2, value2) in nextlabels
            w = get(newlinks, (key,key2), zero(T))
            newlinks[(key,key2)] = w+value2[end]
        end
        queue = merge(queue, _unique_labels(nextstates, T, step+1))
    end

    for (key, value) in newlinks
        src = smap[key[1]]
        dest = smap[key[2]]
        weight = value
        link!(newfsm, src, dest, weight)
    end

    newfsm
end

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

minimize(fsm::FSM{T}) where T = (transpose ∘ determinize ∘ transpose ∘ determinize)(fsm)
