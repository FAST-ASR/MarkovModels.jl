# SPDX-License-Identifier: MIT

#======================================================================
Symbol table
======================================================================#

struct DefaultSymbolTable{T} <: AbstractVector{T}
    size::Int64
end

DefaultSymbolTable(size) = DefaultSymbolTable{Int64}(size)
Base.size(st::DefaultSymbolTable) = (st.size,)
Base.IndexStyle(::Type{<:DefaultSymbolTable}) = IndexLinear()
Base.getindex(::DefaultSymbolTable{T}, i::Int) where T = T(i)

const SymbolTable = Union{AbstractVector, DefaultSymbolTable}

#======================================================================
Matrix-based Finite State Acceptor
======================================================================#

"""
    struct FSA
        α
        T
        ω
        λ
    end

Matrix-based FSA.

# Constructor

    FSA(α, T, ω[, λ])
"""
struct FSA{K,
           TT<:AbstractMatrix{K},
           Tα<:AbstractVector{K},
           Tω<:AbstractVector{K},
           Tλ<:SymbolTable}
    α::Tα
    T::TT
    ω::Tω
    λ::Tλ
end

FSA(α, T, ω) = FSA(α, T, ω, DefaultSymbolTable(size(α, 1)))
nstates(m::FSA) = length(m.α)

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

function _build_T_with_epsilon(arcs, nstates, n_eps, K)
    I_S, J_S, V_S = [], [], K[]
    I_D, J_D, V_D = [], [], K[]
    I_U, J_U, V_U = [], [], K[]
    I_V, J_V, V_V = [], [], K[]
    epsilon_nodes = Set()
    for a in arcs
        if minimum(a.first) > 0
            push!(I_S, a.first[1])
            push!(J_S, a.first[2])
            push!(V_S, a.second)
        elseif a.first[1] <= 0 && a.first[2] <= 0 # arc from and to epsilon node
            push!(I_D, 1 - a.first[1])
            push!(J_D, 1 - a.first[2])
            push!(V_D, a.second)
        elseif a.first[1] <= 0 # outgoing arc from epsilon node
            push!(J_V, 1 - a.first[1])
            push!(I_V, a.first[2])
            push!(V_V, a.second)
        else # incoming arc to epsilon node
            push!(I_U, a.first[1])
            push!(J_U, 1 - a.first[2])
            push!(V_U, a.second)
        end
    end
    S = sparse(I_S, J_S, V_S, nstates, nstates)
    U = sparse(I_U, J_U, V_U, nstates, n_eps)
    V = sparse(I_V, J_V, V_V, nstates, n_eps)
    D = sparse(I_D, J_D, V_D, n_eps, n_eps)
    T = SparseLowRankMatrix(S, D, U, V)
end

function SparseFSA(initws, arcs, finalws, λ = missing)
    # Get the semiring of the FSM.
    K = typeof(initws[1][2])

    # Get the set of states indices.
    states = reduce(
        union,
        [
            Set(map(first, initws)),
            Set(filter(x -> x > 0, map(x -> x.first[1], arcs))),
            Set(filter(x -> x > 0, map(x -> x.first[2], arcs))),
            Set(map(first, finalws))
        ]
    )
    nstates = length(states)

    # Count the epsilon states.
    eps_states = reduce(
        union,
        [
            Set(filter(x -> x <= 0, map(x -> x.first[1], arcs))),
            Set(filter(x -> x <= 0, map(x -> x.first[2], arcs))),
        ]
    )
    n_eps = length(eps_states)

    if n_eps > 0
        T = _build_T_with_epsilon(arcs, nstates, n_eps, K)
    else
        T = _build_T_no_epsilon(arcs, nstates, K)
    end

    FSA(
        sparsevec(map(x -> x[1], initws), map(x -> x[2], initws), nstates),
        T,
        sparsevec(map(x -> x[1], finalws), map(x -> x[2], finalws), nstates),
        ismissing(λ) ? DefaultSymbolTable(nstates) : λ
    )
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

function write_arcs!(file, fsa::FSA{<:Semiring, <:SparseLowRankMatrix})
    n_eps = size(fsa.T.U, 2)
    for iw in (1:n_eps) .+ nstates(fsa)
        write(file, "$i [ label=\"ϵ\" shape=circle style=filled ];\n")
    end

    write_arcs!(file, fsa.T.U; dest_offset = nstates(fsa))
    write_arcs!(file, copy(fsa.T.V'); src_offset = nstates(fsa))
    write_arcs!(file, copy(fsa.T.D); src_offset = nstates(fsa),
                dest_offset = nstates(fsa))
    write_arcs!(file, fsa.T.S)
end

function write_arcs!(file, fsa::FSA{<:Semiring, <:AbstractSparseMatrix})
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

    write_arcs!(file, fsa.T)
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

