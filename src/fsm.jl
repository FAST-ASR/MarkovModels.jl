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

function addstate!(fsm::FSM{T}; pdfindex = nothing, label = nothing) where T
    fsm.idcounter.count += 1
    s = State(fsm.idcounter.count, pdfindex, label, zero(T), zero(T))
    push!(fsm.states, s)
    s
end

"""
    link!(fsm::FSM{T}, src, dest[, weight = zero(T)])

Add a weighted connection between `state1` and `state2`.
"""
function link!(fsm::FSM{T}, src::State{T}, dest::State{T}, weight::T = one(T)) where T
    list = get(fsm.links, src, Link{T}[])
    push!(list, Link{T}(dest, weight))
    fsm.links[src] = list
end

initstates(fsm::FSM{T}) where T = filter(s -> s.startweight ≠ zero(T), fsm.states)
finalstates(fsm::FSM{T}) where T = filter(s -> s.finalweight ≠ zero(T), fsm.states)

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

function Base.show(io, ::MIME"image/svg+xml", fsm::FSM)
    dotpath, dotfile = mktemp()
    svgpath, svgfile = mktemp()

    write(dotfile, "Digraph {\n")
    write(dotfile, "rankdir=LR;")


    for s in states(fsm)
        name = "$(s.id)"
        label = "label=\"$(s.id)"
        attrs = "penwidth=" * (isinit(s) ? "2" : "1")
        label *= isinit(s) ? "/$(round(s.startweight.val, digits = 3))" : ""
        attrs *= " shape=" * (isfinal(s) ? "doublecircle" : "circle")
        label *= isfinal(s) ? "/$(round(s.finalweight.val, digits = 3))\"" : "\""
        write(dotfile, "$name [ $label $attrs ];\n")
    end

    for src in states(fsm)
        for link in links(fsm, src)
            weight = round(link.weight.val, digits = 3)

            srcname = "$(src.id)"
            destname = "$(link.dest.id)"

            ilabel = isnothing(link.ilabel) ? "ϵ" : link.ilabel
            olabel = isnothing(link.olabel) ? "ϵ" : link.olabel
            lname = "$(ilabel):$(olabel)"
            lname *= "/$(weight)"
            write(dotfile, "$srcname -> $destname [ label=\"$(lname)\" ];\n")
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
