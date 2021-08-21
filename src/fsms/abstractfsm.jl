# SPDX-License-Identifier: MIT

#======================================================================
Interface of a generic FSM.
======================================================================#

abstract type AbstractFSM{T<:Semiring} end

"""
    states(fsm)

Return an iterator over the states of the FSM.
"""
states

"""
    arcs(fsm, state)

Return all arcs leaving `state`.
"""
arcs
@deprecate links(fsm, state) arcs(fsm, state)

#======================================================================
Interface of a mutable FSM.
======================================================================#

abstract type AbstractMutableFSM{T} <: AbstractFSM{T} end

"""
    addstate!(fsm::AbstractMutableFSM{T}, label; initweight = zero(T),
              finalweight = zero(T))

Create a state and add it to `fsm`.
"""
addstate!

"""
    addarc!(fsm::AbstractFSM, src, dest, weight)

Add a directed weighted arc from state `src` to state `dest` with
weight `weight`.
"""
addarc!

@deprecate link!(fsm, src, dest) addarc!(fsm, src, dest)
@deprecate link!(fsm, src, dest, weight) addarc!(fsm, src, dest, weight)

#======================================================================
State of a FSM.
======================================================================#

struct State{T<:Semiring}
    id::Int
    label::String
    initweight::T
    finalweight::T
end

"""
    isinit(state)

Return `true` is the state is a starting state.
"""
isinit(state::State{T}) where T = state.initweight ≠ zero(T)
isfinal(state::State{T}) where T = state.finalweight ≠ zero(T)

"""
    isfinal(state)

Return `true` is the state is a final state.
"""
isinit

#======================================================================
Arc of a FSM.
======================================================================#

struct Arc{T<:Semiring}
    dest::State
    weight::T
end

#======================================================================
Pretty display of a FSM.
======================================================================#

function Base.show(io::IO, fsm::AbstractFSM)
    nstates = 0
    narcs = 0
    for s in states(fsm)
        nstates += 1
        for arc in arcs(fsm, s)
            narcs += 1
        end
    end
    print(io, "$(typeof(fsm)) # states: $nstates # arcs: $narcs")
end

function Base.show(io::IO, ::MIME"image/svg+xml", fsm::AbstractFSM)
    dotpath, dotfile = mktemp()
    svgpath, svgfile = mktemp()

    write(dotfile, "Digraph {\n")
    write(dotfile, "rankdir=LR;")

    for s in states(fsm)
        name = "$(s.id)"
        label = s.label
        if isinit(s)
            weight = round(convert(Float64, s.initweight), digits = 3)
            label *= "/$(weight)"
        end
        if isfinal(s)
            weight = round(convert(Float64, s.finalweight), digits = 3)
            label *= "/$(weight)"
        end
        attrs = "shape=" * (isfinal(s) ? "doublecircle" : "circle")
        attrs *= " penwidth=" * (isinit(s) ? "2" : "1")
        attrs *= " label=\"" * label * "\""
        write(dotfile, "$name [ $attrs ];\n")
    end

    for src in states(fsm)
        for arc in arcs(fsm, src)
            weight = round(convert(Float64, arc.weight), digits = 3)
            srcname = "$(src.id)"
            destname = "$(arc.dest.id)"
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
