# SPDX-License-Identifier: MIT

const PdfIndex = Union{Int,Nothing}
const Label = Union{AbstractString,Nothing}

mutable struct State{T<:Semifield}
    id::Int
    initweight::T
    finalweight::T
    pdfindex::PdfIndex
    label::Label
end

isinit(s::State{T}) where T = s.initweight ≠ zero(T)
isfinal(s::State{T}) where T = s.finalweight ≠ zero(T)
islabeled(s::State) = ! isnothing(s.label)
isemitting(s::State)  = ! isnothing(s.pdfindex)
setinit!(s::State{T}, weight::T = one(T)) where T = s.initweight = weight
setfinal!(s::State{T}, weight::T = one(T)) where T = s.finalweight = weight

mutable struct Link{T<:Semifield}
    dest::State
    weight::T
end

"""
    struct FSM{T<:Semifield}
        states # vector of states
        links # Dict state -> vector of links
    end

Probabilistic finite state machine.
"""
struct FSM{T<:Semifield}
    states::Vector{State{T}}
    links::Dict{State, Vector{Link{T}}}
end
FSM{T}() where T = FSM{T}(State{T}[], Dict{State, Vector{Link{T}}}())
FSM() = FSM{LogSemifield{Float64}}()

states(fsm::FSM) = fsm.states
links(fsm::FSM{T}, state::State{T}) where T = get(fsm.links, state, Link{T}[])

function addstate!(fsm::FSM{T}; initweight = zero(T), finalweight = zero(T),
                   pdfindex = nothing, label = nothing) where T
    s = State(length(fsm.states)+1, initweight, finalweight, pdfindex, label)
    push!(fsm.states, s)
    s
end

function link!(fsm::FSM{T}, src::State{T}, dest::State{T}, weight::T = one(T)) where T
    list = get(fsm.links, src, Link{T}[])
    link = Link{T}(dest, weight)
    push!(list, link)
    fsm.links[src] = list
    link
end

function Base.show(io, ::MIME"image/svg+xml", fsm::FSM)
    dotpath, dotfile = mktemp()
    svgpath, svgfile = mktemp()

    write(dotfile, "Digraph {\n")
    write(dotfile, "rankdir=LR;")

    for s in states(fsm)
        name = "$(s.id)"
        label = islabeled(s) ? "$(s.label)" : "ϵ"
        label *= isemitting(s) ? ":$(s.pdfindex)" : ":ϵ"
        attrs = "shape=" * (isfinal(s) ? "doublecircle" : "circle")
        attrs *= " penwidth=" * (isinit(s) ? "2" : "1")
        attrs *= " label=\"" * label * "\""
        attrs *= " style=filled fillcolor=" * (isemitting(s) ? "lightblue" : "none")
        write(dotfile, "$name [ $attrs ];\n")
    end

    for src in states(fsm)
        for link in links(fsm, src)
            weight = round(convert(Float64, link.weight), digits = 3)
            srcname = "$(src.id)"
            destname = "$(link.dest.id)"
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

#######################################################################
# FSM operations

"""
    renormalize!(fsm)

Ensure the that all the weights of all the outgoing arcs leaving a
state sum up to 1.
"""
function renormalize!(fsm::FSM{T}) where T
    total = zero(T)
    for s in filter(isinit, states(fsm)) total += s.initweight end
    for s in filter(isinit, states(fsm)) s.initweight /= total end

    total = zero(T)
    for s in filter(isfinal, states(fsm)) total += s.finalweight end
    for s in filter(isfinal, states(fsm)) s.finalweight /= total end

    for src in states(fsm)
        total = zero(T)
        for link in links(fsm, src) total += link.weight end
        for link in links(fsm, src) link.weight /= total end
    end

    fsm
end

"""
    replace(fsm, subfsms)

Replace the state in `fsm` wiht a sub-fsm from `subfsms`. The pairing
is done with label of the state, i.e. the state with label `l` will be
replaced by `subfsms[l]`. States that don't have matching labels are
left untouched.

You can specify
"""
function Base.replace(fsm::FSM{T}, subfsms::Dict; matching_func=last, delim="!") where T
    newfsm = FSM{T}()

    smap_in = Dict()
    smap_out = Dict()
    for s in states(fsm)
        last_label = isnothing(s.label) ? nothing : (matching_func∘split)(s.label, delim)
        if last_label in keys(subfsms)
            smap = Dict()
            for cs in states(subfsms[last_label])
                label = "$(s.label)$delim$(cs.label)"
                ns = addstate!(newfsm, pdfindex = cs.pdfindex, label = label,
                               initweight = s.initweight * cs.initweight,
                               finalweight = s.finalweight * cs.finalweight)
                smap[cs] = ns

                if isinit(cs) smap_in[s] = ns end
                if isfinal(cs) smap_out[s] = ns end
            end

            for cs in states(subfsms[last_label])
                for link in links(subfsms[last_label], cs)
                    link!(newfsm, smap[cs], smap[link.dest], link.weight)
                end
            end

        else
            ns = addstate!(newfsm, pdfindex = s.pdfindex, label = s.label,
                           initweight = s.initweight, finalweight = s.finalweight)
            smap_in[s] = ns
            smap_out[s] = ns
        end
    end

    for osrc in states(fsm)
        for link in links(fsm, osrc)
            src = smap_out[osrc]
            dest = smap_in[link.dest]
            link!(newfsm, src, dest, link.weight)
        end
    end

    newfsm
end

function _unique_labels(statelist, T, step)
    labels = Dict()
    for (s, w) in statelist
        lstates, iw, fw, tw = get(labels, (s.label, step), (Set(), zero(T), zero(T), zero(T)))
        push!(lstates, s)
        labels[(s.label, step)] = (lstates, iw+s.initweight, fw+s.finalweight, tw+w)
    end

    # Inverse the map so that the set of states is the key.
    retval = Dict()
    for (key, value) in labels
        retval[value[1]] = (key[1], value[2], value[3], value[4])
    end
    retval
end

"""
    determinize(fsm)

Determinize the FSM w.r.t. the state labels.
"""
function determinize(fsm::FSM{T}) where T
    newfsm = FSM{T}()
    smap = Dict()
    newlinks = Dict()

    initstates = [(s, zero(T)) for s in filter(isinit, collect(states(fsm)))]
    queue = _unique_labels(initstates, T, 0)
    while ! isempty(queue)
        key, value = pop!(queue)
        lstates = key
        label, iw, fw, tw = value
        step = 0

        if key ∉ keys(smap)
            s = addstate!(newfsm, label = label, initweight = iw, finalweight = fw)
            smap[key] = s
        end

        nextstates = []
        for ls in lstates
            for link in links(fsm, ls)
                push!(nextstates, (link.dest, link.weight))
            end
        end

        nextlabels = _unique_labels(nextstates, T, step+1)
        for (key2, value2) in nextlabels
            w = get(newlinks, (key,key2), zero(T))
            newlinks[(key,key2)] = w+value2[end]
        end
        queue = merge(queue, _unique_labels(nextstates, T, step+1))
    end

    for (key, value) in newlinks
        src = smap[key[1]]
        dest = smap[key[2]]
        weight = value
        link!(newfsm, src, dest, weight)
    end

    newfsm
end

"""
    transpose(fsm)

Reverse the direction of the arcs.
"""
function Base.transpose(fsm::FSM{T}) where T
    newfsm = FSM{T}()
    smap = Dict()
    for s in states(fsm)
        ns = addstate!(newfsm, label = s.label, initweight = s.finalweight,
                       finalweight = s.initweight, pdfindex = s.pdfindex)
        smap[s] = ns
    end

    for src in states(fsm)
        for link in links(fsm, src)
            link!(newfsm, smap[link.dest], smap[src], link.weight)
        end
    end

    newfsm
end

"""
    minimize(fsm)

Return a minimal equivalent fsm.
"""
minimize(fsm::FSM{T}) where T = (transpose ∘ determinize ∘ transpose ∘ determinize)(fsm)

function _calculate_distances(ω::AbstractVector{T}, A::AbstractMatrix{T}) where T
    Aᵀ = transpose(A)
    I = findall(ω .> zero(T))
    queue = Set{Tuple{Int,Int}}([(state, 0) for state in I])
    visited = Set{Int}(I)
    distances = zeros(Int, length(ω))
    while ! isempty(queue)
        state, dist = pop!(queue)
        for nextstate in findall(Aᵀ[state,:] .> zero(T))
            if nextstate ∉ visited
                push!(queue, (nextstate, dist + 1))
                push!(visited, nextstate)
                distances[nextstate] = dist + 1
            end
        end
    end
    distances
end

"""
    union(fsm1, fsm2, ...)

Merge several FSMs into a single one.
"""
function Base.union(fsm1::FSM{T}, fsm2::FSM{T}) where T
    fsm = FSM{T}()
    s = addstate!(fsm)
    s1 = addstate!(fsm, label = "#fsm1")
    s2 = addstate!(fsm, label = "#fsm2")
    link!(fsm, s, s1)
    link!(fsm, s, s2)
    setinit!(s)
    setfinal!(s1)
    setfinal!(s2)
    unionfsm = replace(fsm, Dict("#fsm1" => fsm1, "#fsm2" => fsm2))
    foreach(states(unionfsm)) do s
        if ! isnothing(s.label)
            s.label = replace(s.label, "#fsm1!" => "")
            s.label = replace(s.label, "#fsm2!" => "")
        end
    end
    return unionfsm
end
Base.union(fsm::FSM{T}, rest::FSM{T}...) where T = foldl(union, rest, init=fsm)

"""
    compile(fsm; allocator = spzeros)

Compile `fsm` into a inference-friendly format. `allocator` is a
function analogous to `zeros` which create a matrix and fill with
zero elements.
"""
function compile(fsm::FSM{T}; allocator = spzeros) where T
    allstates = collect(states(fsm))
    S = length(allstates)

    # Initial states' probabilities.
    π = allocator(T, S)
    for s in filter(isinit, allstates) π[s.id] = s.initweight end

    # Final states' probabilities.
    ω = allocator(T, S)
    for s in filter(isfinal, allstates) ω[s.id] = s.finalweight end

    # Transition matrix.
    A = allocator(T, S, S)
    Aᵀ = allocator(T, S, S)
    for src in allstates
        for link in links(fsm, src)
            A[src.id, link.dest.id] = link.weight
            Aᵀ[link.dest.id, src.id] = link.weight
        end
    end

    # For each state the distance to the nearest final state.
    dists = _calculate_distances(ω, A)

    # Pdf index mapping.
    pdfmap = [s.pdfindex for s in sort(allstates, by = p -> p.id)]

    (π = π, ω = ω, A = A, Aᵀ = Aᵀ, dists = dists, pdfmap = pdfmap)
end

"""
    gpu(cfsm)

Move the compiled fsm `cfsm` to a GPU.
"""
function gpu(cfsm) where T
    if ! issparse(cfsm.π)
        return (
            π = CuArray(cfsm.π),
            ω = CuArray(cfsm.ω),
            A = CuArray(cfsm.A),
            Aᵀ = CuArray(cfsm.Aᵀ),
            dists = cfsm.dists,
            pdfmap = cfsm.pdfmap
        )
    end

    A = CuSparseMatrixCSC(cfsm.A)
    Aᵀ = CuSparseMatrixCSC(cfsm.Aᵀ)
    return (
        π = CuSparseVector(cfsm.π),
        ω = CuSparseVector(cfsm.ω),
        A = CuSparseMatrixCSR(Aᵀ.colPtr, Aᵀ.rowVal, Aᵀ.nzVal, A.dims),
        Aᵀ = CuSparseMatrixCSR(A.colPtr, A.rowVal, A.nzVal, A.dims),
        dists = cfsm.dists,
        pdfmap = cfsm.pdfmap
    )
end
