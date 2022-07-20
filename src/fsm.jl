# SPDX-License-Identifier: MIT

const LabelMonoid = SequenceMonoid
Label() = one(LabelMonoid)
Label(x) = LabelMonoid(tuple(x))

const TransitionMatrix{K} =
    Union{AbstractSparseMatrix{K}, SparseLowRankMatrix{K}} where K

struct FSM{K<:Semiring,L}
    # Initial weight vectors extended with a final state with initial
    # initial weight of zero.
    α̂::AbstractSparseVector{K}

    # Transition matrix T extended with a "final" state. All the
    # edges to the final state encode the "ω" vector.
    T̂::TransitionMatrix{K}

    λ::AbstractVector{L}
end

function FSM(α::AbstractSparseVector, T::TransitionMatrix,
             ω::AbstractSparseVector, λ::AbstractVector)

    Tω = hcat(T, ω)
    p = fill!(similar(ω, length(ω) + 1), zero(eltype(ω)))
    p[end] = one(eltype(ω))
    T̂ = vcat(Tω, reshape(p, 1, :))

    FSM(vcat(α, zero(eltype(α))), T̂, λ)
end

function Base.getproperty(fsm::FSM, sym::Symbol)
    if sym === :α
        return fsm.α̂[1:end-1]
    elseif sym === :ω
        return fsm.T̂[1:end-1, end]
    elseif sym === :T
        return fsm.T̂[1:end-1, 1:end-1]
    else
        return getfield(fsm, sym)
    end
end

function Adapt.adapt_structure(::Type{<:CuArray}, fsm::FSM)
    FSM(
        CuSparseVector(fsm.α̂),
        CuSparseMatrixCSR(CuSparseMatrixCSC(fsm.T̂)),
        fsm.λ
    )
end

function FSM(initws, arcs, finalws, λ)
    # Get the set of states indices.
    states = (Set(map(first, initws))
              ∪ Set(map(first, finalws))
              ∪ Set(vcat([collect(tup) for tup in map(first, arcs)]...)))
    nstates = size(λ, 1)
    K = typeof(initws[1][2])

    if size(arcs, 1) > 0
        # Check if there are epsilon nodes.
        nodes = Set()
        for a in arcs
            push!(nodes, a.first...)
        end
        has_epsilons = minimum(nodes) > 0 ? false : true

        if ! has_epsilons
            T = sparse(map(x -> x[1][1], arcs),
                       map(x -> x[1][2], arcs),
                       map(x -> x[2], arcs), nstates, nstates)
        else
            I_S, J_S, V_S = [], [], K[]
            I_U, J_U, V_U = [], [], K[]
            I_V, J_V, V_V = [], [], K[]
            for a in arcs
                if minimum(a.first) > 0
                    push!(I_S, a.first[1])
                    push!(J_S, a.first[2])
                    push!(V_S, a.second)
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
            U = sparse(I_U, J_U, V_U)
            V = sparse(I_V, J_V, V_V)
            T = SparseLowRankMatrix(S, U, V)
        end
    else
        T = spzeros(K, nstates, nstates)
    end

    FSM(
        sparsevec(map(x -> x[1], initws), map(x -> x[2], initws), nstates),
        T,
        sparsevec(map(x -> x[1], finalws), map(x -> x[2], finalws), nstates),
        λ
    )
end

function FSM(s::AbstractString)
    data = JSON.parse(s)
    K = eval(Meta.parse(data["semiring"]))
    FSM(
        [a => K(b) for (a, b) in data["initstates"]],
        [(a, b) => K(c) for (a, b, c) in data["arcs"]],
        [a => K(b) for (a, b) in data["finalstates"]],
        [Label(a) for a in data["labels"]]
    )
end

nstates(m::FSM) = length(m.α)

function arcs(T::AbstractSparseMatrix)
    I, J, V = findnz(T)
    retval = []
    for (i, j, v) in zip(I, J, V)
        push!(retval, (i, j, v))
    end
    retval
end

function arcs(T::AbstractMatrix)
    retval = []
    for (i, j) in zip(1:size(T, 1), 1:size(T, 2))
        push!(retval, (i, j, T[i, j]))
    end
    retval
end

#======================================================================
SVG display of FSM
======================================================================#

showlabel(label) = join(val(label), ":")

function Base.show(io::IO, ::MIME"image/svg+xml", fsm::FSM)
    # Move the FSM to the CPU memory.
    if typeof(fsm.T) <: AbstractCuSparseMatrix
        fsm = adapt(Array, fsm)
    end

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

    if fsm.T isa SparseLowRankMatrix
        for i in (1:size(fsm.T.U, 2)) .+ nstates
            write(dotfile, "$i [ label=\"ϵ\" shape=circle style=filled ];\n")
        end
        S = fsm.T.S

        for (i, j, w) in arcs(fsm.T.U)
            weight = round(convert(Float64, w.val), digits = 3)
            srcname = "$i"
            destname = "$(nstates + j)"
            write(dotfile, "$srcname -> $destname [ label=\"$(weight)\" ];\n")
        end

        for (i, j, w) in arcs(fsm.T.V)
            weight = round(convert(Float64, w.val), digits = 3)
            srcname = "$(nstates + j)"
            destname = "$i"
            write(dotfile, "$srcname -> $destname [ label=\"$(weight)\" ];\n")
        end
    else
        S = fsm.T
    end

    for (i, j, w) in arcs(S)
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
