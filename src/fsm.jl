# SPDX-License-Identifier: MIT

struct FSM{K<:Semiring}
    α::AbstractVector{K}
    T::AbstractMatrix{K}
    ω::AbstractVector{K}
    λ::AbstractVector{UnionConcatSemiring}
end

#======================================================================
SVG display of FSM
======================================================================#

function arcs(T::AbstractArray)
    retval = []
    for i in 1:size(T, 1)
        for j in 1:size(T, 2)
            if ! iszero(T[i, j])
                push!(retval, (i, j, T[i,j]))
            end
        end
    end
    retval
end

function showlabel(label)
    retval  = ""
    for (i, seq) in enumerate(sort(collect(label.val)))
        if i > 1 retval *= ":" end
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
