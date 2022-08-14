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

function write_arc!(io::IO, src, dest, label, weight)
    wstr = @sprintf "%.3g" weight
    write(io, "$src -> $dest [ label=\"$label/$wstr\" ];\n")

end

function write_states!(io::IO, fsa::FSA)
    write(io, "0 [ shape=\"point\" ];\n")
    write(io, "$(nstates(fsa) + 1) [ shape=\"point\" ];\n")

    penwidth = "1"
    shape = "circle"
    for i in 1:nstates(fsa)
        name = "$i"
        label = "$i"

        attrs = "shape=" * shape
        attrs *= " penwidth=" * penwidth
        attrs *= " label=\"" * label * "\""
        write(io, "$name [ $attrs ];\n")
    end

    for i in 1:nstates(fsa)
        if ! iszero(fsa.α[i])
            write_arc!(io, 0, i, fsa.λ[i], val(fsa.α[i]))
        end

        if ! iszero(fsa.ω[i])
            write_arc!(io, i, nstates(fsa) + 1, fsa.λ[i], val(fsa.ω[i]))
        end
    end
end

function write_arcs!(io::IO, fsa::FSA)
    T = fsa.T
    for i = 1:size(T, 1), j = 1:size(T, 2)
        iszero(T[i, j]) && continue
        write_arc!(io, i, j, fsa.λ[j], val(T[i, j]))
    end
end

function Base.show(io::IO, ::MIME"image/dot", fsa::FSA)
    write(io, "Digraph {\n")
    write(io, "rankdir=LR;")
    write_states!(io, fsa)
    write_arcs!(io, fsa)
    write(io, "}\n")
end

function Base.show(io::IO, ::MIME"image/svg+xml", fsa::FSA)
    dotpath, dotfile = mktemp()
    try
        show(dotfile, MIME("image/dot"), fsa)
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
