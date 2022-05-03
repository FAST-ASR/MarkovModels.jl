# SPDX-License-Identifier: MIT

"""
    Label(x)

Create a FSM label. A label is an element the Union-Concatenation
Semiring, i.e. it is a set of sequences.
"""
Label(x) = UnionConcatSemiring(Set([SymbolSequence([x])]))

struct FSM{K<:Semiring}
    α::AbstractSparseVector{K}
    T::AbstractSparseMatrix{K}
    ω::AbstractSparseVector{K}
    λ::AbstractVector{UnionConcatSemiring}
end

function FSM(nstates, initws, arcs, finalws, λ)
    α = sparsevec(map(x -> x[1], initws), map(x -> x[2], initws), nstates)
    T = sparse(map(x -> x[1][1], arcs), map(x -> x[1][2], arcs),
               map(x -> x[2], arcs), nstates, nstates)
    ω = sparsevec(map(x -> x[1], finalws), map(x -> x[2], finalws), nstates)
    FSM(α, T, ω, λ)
end

nstates(m::FSM) = length(m.α)
arcs(T::AbstractSparseArray) = zip(findnz(T)...)

#======================================================================
SVG display of FSM
======================================================================#



function showlabel(label)
    retval  = ""
    for (i, seq) in enumerate(sort(collect(label.val)))
        retval *= join(seq)
    end
    retval
end

function Base.show(io::IO, ::MIME"image/svg+xml", fsm::FSM{T}) where T
    dotpath, dotfile = mktemp()
    svgpath, svgfile = mktemp()

    write(dotfile, "Digraph {\n")
    write(dotfile, "rankdir=LR;")

    nstates = length(fsm.α)

    for i in 1:nstates
        name = "$i"
        label = showlabel(fsm.λ[i])

        penwidth = "1"
        if ! iszero(fsm.α[i])
            weight = round(convert(Float64, fsm.α[i].val), digits = 3)
            label *= "/$(weight)"
            penwidth = "2"
        else
            penwidth = "1"
       end

        if ! iszero(fsm.ω[i])
            weight = round(convert(Float64, fsm.ω[i].val), digits = 3)
            label *= "/$(weight)"
            shape = "doublecircle"
        else
            shape = "circle"
        end

        attrs = "shape=" * shape
        attrs *= " penwidth=" * penwidth
        attrs *= " label=\"" * label * "\""
        write(dotfile, "$name [ $attrs ];\n")
    end

    for (i, j, w) in arcs(fsm.T)
        weight = round(convert(Float64, w.val), digits = 3)
        srcname = "$i"
        destname = "$j"
        write(dotfile, "$srcname -> $destname [ label=\"$(weight)\" ];\n")
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
