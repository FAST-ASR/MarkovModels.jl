# SPDX-License-Identifier: MIT

const LabelMonoid = SequenceMonoid
Label() = one(LabelMonoid)
Label(x) = LabelMonoid(tuple(x))

struct FSM{K<:Semiring,L}
    # Initial weight vectors extended with a final state with initial
    # initial weight of zero.
    α̂::AbstractSparseVector{K}

    # Transition matrix T extended with a "final" state. All the
    # edges to the final state encode the "ω" vector.
    T̂::AbstractSparseMatrix{K}

    λ::AbstractVector{L}
end

function FSM(α::AbstractSparseVector, T::AbstractSparseMatrix,
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

function Adapt.adapt_structure(to, fsm::FSM)
    FSM(
        adapt(to, fsm.α̂),
        adapt(to, fsm.T̂),
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
        T = sparse(map(x -> x[1][1], arcs),
                   map(x -> x[1][2], arcs),
                   map(x -> x[2], arcs), nstates, nstates)
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

function arcs(T::AbstractSparseArray)
    I, J, V = findnz(T)
    retval = []
    for (i, j, v) in zip(I, J, V)
        push!(retval, (i, j, v))
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
