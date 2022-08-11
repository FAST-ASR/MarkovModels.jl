# SPDX-License-Identifier: MIT

"""
    struct FSA
        α
        T
        ω
        λ
    end

Matrix-based Finite State Automata.

# Constructor

    FSA(α, T, ω, λ)
"""
struct FSA{K,
           TT<:TransitionMatrix{K},
           Tα<:StateVector{K},
           Tω<:StateVector{K},
           Tλ<:AbstractVector}
    α::Tα
    T::TT
    ω::Tω
    λ::Tλ
end


"""
    nstates(fsa)

Returns the number of states in `fsa` (not including epsilon states).
"""
nstates(fsa::FSA) = length(fsa.α)


"""
    narcs(fsa)

Returns the number of arcs in `fsa`.
"""
narcs(fsa::FSA) = sum(findall(.! iszero.(fsa.T)))


#======================================================================
FSA construction
======================================================================#

function _build_T_no_epsilon(arcs, nstates, K)
    if size(arcs, 1) > 0
        T = sparse(
            map(x -> x[1][1], arcs),
            map(x -> x[1][2], arcs),
            map(x -> x.second, arcs),
            nstates,
            nstates
        )
    else
        T = spzeros(K, nstates, nstates)
    end
    T
end

#======================================================================
SVG display of FSA
======================================================================#

function write_states!(file, fsa::FSA)
    write(file, "0 [ shape=\"point\" ];\n")
    write(file, "$(nstates(fsa) + 1) [ shape=\"point\" ];\n")

    penwidth = "1"
    shape = "circle"
    for i in 1:nstates(fsa)
        name = "$i"
        label = "$(fsa.λ[i])"

        attrs = "shape=" * shape
        attrs *= " penwidth=" * penwidth
        attrs *= " label=\"" * label * "\""
        write(file, "$name [ $attrs ];\n")
    end

    for i in 1:nstates(fsa)
        if ! iszero(fsa.α[i])
            weight = round(convert(Float64, val(fsa.α[i])), digits = 3)
            write(file, "0 -> $i [ label=\"$(weight)\" ];\n")
        end

        if ! iszero(fsa.ω[i])
            weight = round(convert(Float64, val(fsa.ω[i])), digits = 3)
            write(file, "$i -> $(nstates(fsa) + 1) [ label=\"$(weight)\" ];\n")
        end
    end
end

function write_arcs!(file, T::AbstractMatrix; src_offset = 0, dest_offset = 0)
    for i = 1:size(T, 1), j = 1:size(T, 2)
        iszero(T[i, j]) && continue
        weight = round(convert(Float64, val(T[i, j])), digits = 3)
        srcname = "$(i + src_offset)"
        destname = "$(j + dest_offset)"
        write(file, "$srcname -> $destname [ label=\"$(weight)\" ];\n")
    end
end

function write_arcs!(file, fsa::FSA)
    write_arcs!(file, fsa.T)
end


function Base.show(io::IO, ::MIME"dot", fsa::FSA)
    dotfile = io
    write(dotfile, "Digraph {\n")
    write(dotfile, "rankdir=LR;")

    write_states!(dotfile, fsa)
    write_arcs!(dotfile, fsa)

    write(dotfile, "}\n")
    flush(dotfile)

    svgpath, svgfile = mktemp()
    dotstr = read(dotfile, String)
    write(io, dotstr)
end

function Base.show(io::IO, ::MIME"image/svg+xml", fsa::FSA)
    dotpath, dotfile = mktemp()
    try
        write(dotfile, "Digraph {\n")
        write(dotfile, "rankdir=LR;")

        write_states!(dotfile, fsa)
        write_arcs!(dotfile, fsa)

        write(dotfile, "}\n")
        flush(dotfile)

        svgpath, svgfile = mktemp()
        try
            run(`dot -Tsvg $(dotpath) -o $(svgpath)`)
            xml = read(svgfile, String)
            write(io, xml)
        finally
            close(svgfile)
            rm(svgpath)
        end

    finally
        close(dotfile)
        rm(dotpath)
    end
end
