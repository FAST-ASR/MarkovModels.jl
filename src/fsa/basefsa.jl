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

    FSA(α, T, ω, λ[, properties])
"""
struct FSA{K,
           TT<:TransitionMatrix{K},
           Tα<:StateVector{K},
           Tω<:StateVector{K},
           Tλ<:AbstractVector,
           Tprops<:Properties}
    α::Tα
    T::TT
    ω::Tω
    λ::Tλ
    properties::Tprops
end

FSA(α, T, ω, λ; kwargs...) = FSA(α, T, ω, λ, Properties(; kwargs...))

"""
    fsatype(fsa)

Return a string describing the type of `fsa`.
"""
fsatype(fsa::FSA)


"""
    nstates(fsa)

Returns the number of states in `fsa` (not including epsilon states).
"""
nstates(fsa::FSA) = length(fsa.α)


"""
    narcs(fsa)

Returns the number of arcs in `fsa`.
"""
narcs(fsa::FSA) = sum( .! iszero.(fsa.α) ) + sum( .! iszero.(fsa.T) ) + sum( .! iszero(fsa.ω) )


#======================================================================
Summary display of the FSA
======================================================================#

showproperty(props, proptype) = props isa proptype ? "yes" : "???"

function Base.summary(fsa::FSA{K}) where K
    retval = """
    | Property          | Value                                                               |
    |:------------------|:--------------------------------------------------------------------|
    | fsa type          | $(fsatype(fsa))                                                     |
    | semiring          | $K                                                                  |
    | # states          | $(nstates(fsa))                                                     |
    | # arcs            | $(narcs(fsa))                                                       |
    """

    tags = ""
    if fsa.properties isa HasAccessibleProperty
        tags *= "accessible "
    end
    if fsa.properties isa HasAcyclicProperty
        tags *= "acyclic "
    end
    if fsa.properties isa HasCoAccessibleProperty
        tags *= "coaccessible "
    end
    if fsa.properties isa HasDeterministicProperty
        tags *= "deterministic "
    end
    if fsa.properties isa HasLexicographicallySortedProperty
        tags *= "lexsorted "
    end
    if fsa.properties isa HasNormalizedProperty
        tags *= "normalized "
    end
    if fsa.properties isa HasTopologicallySortedProperty
        tags *= "topsorted "
    end
    if fsa.properties isa HasWeightPropagatedProperty
        tags *= "weightpropagated "
    end

    Markdown.parse(retval * "| tags | $tags |")
end


#======================================================================
SVG display of FSA
======================================================================#

function write_arc!(io::IO, src, dest, weight)
    wstr = @sprintf "%.3g" weight
    write(io, "$src -> $dest [ label=\"$wstr\" ];\n")

end

function write_arc!(io::IO, src, dest, label, weight)
    wstr = @sprintf "%.3g" weight
    write(io, "$src -> $dest [ label=\"$label/$wstr\" ];\n")

end

function write_states!(io::IO, fsa::FSA, colors = nothing)
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

        if ! isnothing(colors) && i ∈ keys(colors)
            attrs *= " style=filled fillcolor=\"$(colors[i])\""
        end
        write(io, "$name [ $attrs ];\n")
    end

    for i in 1:nstates(fsa)
        if ! iszero(fsa.α[i])
            write_arc!(io, 0, i, fsa.λ[i], val(fsa.α[i]))
        end

        if ! iszero(fsa.ω[i])
            write_arc!(io, i, nstates(fsa) + 1, val(fsa.ω[i]))
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

function Base.show(io::IO, ::MIME"image/dot", fsa_colors::Tuple{<:FSA, <:AbstractDict})
    fsa, colors = fsa_colors
    write(io, "Digraph {\n")
    write(io, "rankdir=LR;")
    write_states!(io, fsa, colors)
    write_arcs!(io, fsa)
    write(io, "}\n")
end

function Base.show(io::IO, mime::MIME"image/svg+xml",
                   fsa_colors::Union{FSA, Tuple{<:FSA, <:AbstractDict}})
    dotpath, dotfile = mktemp()
    try
        show(dotfile, MIME("image/dot"), fsa_colors)
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

