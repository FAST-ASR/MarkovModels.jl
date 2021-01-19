# Implementation of common FSM operations.

#######################################################################
# FSM transpose

"""
    Base.transpose(fsm)

Transpose the fsm, i.e. reverse all it's arcs. The final state becomes
the initial state.
"""
function Base.transpose(fsm::FSM{T}) where T
    nfsm = FSM{T}()
    smap = Dict(initstateid => finalstate(nfsm), finalstateid => initstate(nfsm))
    for s in states(fsm)
        (isinit(s) || isfinal(s)) && continue
        smap[s.id] = addstate!(nfsm, id = s.id, pdfindex = s.pdfindex, label = s.label)
    end

    for l in links(fsm)
        link!(smap[l.dest.id], smap[l.src.id], l.weight)
    end
    nfsm
end

#######################################################################
# Union of FSMs

"""
    union(fsm1, fsm2, ...)

Merge several FSMs into a single one.

# Examples
```julia-repl
julia> fsm1 = LinearFSM(["a", "b", "c"], Dict("a"=>1))
julia> fsm2 = LinearFSM(["a", "d", "c"], Dict("a"=>1))
julia> union(fsm1, fsm2)
```
Input:

  * `fsm1`

    ![See the online documentation to visualize the image](images/union_input1.svg)
  * `fsm2`

    ![See the online documentation to visualize the image](images/union_input2.svg)

Output:

![See the online documentation to visualize the image](images/union_output.svg)

"""
function Base.union(fsm1::AbstractFSM{T}, fsm2::AbstractFSM{T}) where T
    fsm = FSM{T}()

    smap = Dict{State, State}(initstate(fsm1) => initstate(fsm),
                              finalstate(fsm1) => finalstate(fsm))
    for s in states(fsm1)
        if s.id == finalstateid || s.id == initstateid continue end
        smap[s] = addstate!(fsm, pdfindex = s.pdfindex, label = s.label)
    end
    for l in links(fsm1) link!(smap[l.src], smap[l.dest], l.weight) end

    smap = Dict{State, State}(initstate(fsm2) => initstate(fsm),
                              finalstate(fsm2) => finalstate(fsm))
    for s in states(fsm2)
        if s.id == finalstateid || s.id == initstateid continue end
        smap[s] = addstate!(fsm, pdfindex = s.pdfindex, label = s.label)
    end
    for l in links(fsm2) link!(smap[l.src], smap[l.dest], l.weight) end

    fsm
end
Base.union(fsm::AbstractFSM, rest::AbstractFSM...) = foldl(union, rest, init=fsm)

#######################################################################
# Concatenation

"""
    concat(fsm1, fsm2, ...)

Concatenate several FSMs into single FSM.

# Examples
```julia-repl
julia> fsm1 = LinearFSM(["a", "b"])
julia> fsm2 = LinearFSM(["c", "d"])
julia> fsm3 = LinearFSM(["e"])
julia> concat(fsm1, fsm2, fsm3)
```
Input:
  * `fsm1`
  ![See the online documentation to visualize the image](images/concat_input1.svg)
  * `fsm2`
  ![See the online documentation to visualize the image](images/concat_input2.svg)
  * `fsm3`
  ![See the online documentation to visualize the image](images/concat_input3.svg)

Output:
  ![See the online documentation to visualize the image](images/concat_output.svg)
"""
function concat(fsm1::AbstractFSM{T}, fsm2::AbstractFSM{T}) where T
    fsm = FSM{T}()

    cs = addstate!(fsm) # special non-emitting state for concatenaton
    smap = Dict(initstate(fsm1) => initstate(fsm), finalstate(fsm1) => cs)
    for s in states(fsm1)
        (isinit(s) || isfinal(s)) && continue
        smap[s] = addstate!(fsm, pdfindex = s.pdfindex, label = s.label)
    end
    for l in links(fsm1) link!(smap[l.src], smap[l.dest], l.weight) end

    smap = Dict(initstate(fsm2) => cs, finalstate(fsm2) => finalstate(fsm))
    for s in states(fsm2)
        (isinit(s) || isfinal(s)) && continue
        smap[s] = addstate!(fsm, pdfindex = s.pdfindex, label = s.label)
    end
    for l in links(fsm2) link!(smap[l.src], smap[l.dest], l.weight) end

    fsm
end
concat(fsm1::AbstractFSM, rest::AbstractFSM...) = foldl(concat, rest, init=fsm1)

#######################################################################
# Weight normalization

"""
    weightnormalize(fsm)

Change the weight of the links such that the sum of the exponentiated
weights of the outgoing links from one state will sum up to one.

# Examples
```julia-repl
julia> fsm = union(LinearFSM(["a", "b"]), LinearFSM(["c", "d"]))
julia> for s in states(fsm)
    if ! isinit(s) && ! isfinal(s)
        link!(fsm, s, s)
    end
end
julia> fsm |> weightnormalize!
```
Input:

![See the online documentation to visualize the image](images/wnorm_input.svg)

Output:

![See the online documentation to visualize the image](images/wnorm_output.svg)

!!! note
    This function has the side effect to "determinize" the FSM, that is,
    the resulting FSM will have at most one arc between each pair of
    node.

"""
function weightnormalize(fsm::AbstractFSM{T}) where T
    nfsm = FSM{T}()
    smap = Dict(
        initstateid => initstate(nfsm),
        finalstateid => finalstate(nfsm)
    )
    for s in states(fsm)
        (isinit(s) || isfinal(s)) && continue
        smap[s.id] = addstate!(nfsm, pdfindex = s.pdfindex, label = s.label)
    end

    for s in states(fsm)
        total = -Inf
        for l in links(s) total = logaddexp(total, l.weight) end
        for l in links(s)
            link!(smap[l.src.id], smap[l.dest.id], l.weight - total)
        end
    end
    nfsm
end

#######################################################################
# FSM determinization

"""
    determinize!(fsm)

Transform `fsm` such that each state has at most one link to any other
states.
"""
function determinize(fsm::AbstractFSM{T}) where T
    nfsm = FSM{T}()
    newstates = Dict(
        initstate(fsm) => initstate(nfsm),
        finalstate(fsm) => finalstate(nfsm)
    )
    for state in states(fsm)
        (isfinal(state) || isinit(state)) && continue
        newstates[state] = addstate!(nfsm, pdfindex = state.pdfindex, label = state.label)
    end

    newlinks = Dict()
    for link in links(fsm)
        key = (link.src, link.dest)
        w₀ = get(newlinks, key, -Inf)
        newlinks[key] = logaddexp(w₀, link.weight)
    end

    for key in keys(newlinks)
        src, dest = key
        weight = newlinks[key]
        link!(newstates[src], newstates[dest], weight)
    end

    nfsm
end

#######################################################################
# FSM minimization

# propagate the weight of each link through the graph
function _distribute(fsm::AbstractFSM{T}) where T
    nfsm = FSM{T}()

    newstates = Dict(initstate(fsm) => initstate(nfsm),
                     finalstate(fsm) => finalstate(nfsm))
    for state in states(fsm)
        (isinit(state) || isfinal(state)) && continue
        newstates[state] = addstate!(nfsm, pdfindex = state.pdfindex, label = state.label)
    end

    stack = [(initstate(fsm), 0.0)]
    while ! isempty(stack)
        state, weight = popfirst!(stack)

        for link in links(state)
            link!(newstates[state], newstates[link.dest], weight + link.weight)
            push!(stack, (link.dest, weight + link.weight))
        end
    end

    nfsm
end

function _leftminimize(fsm::AbstractFSM{T}) where T
    # 1. Build the tree of string generated by the FSM
    tree = Dict()
    stack = [(initstate(fsm), tree)]
    while ! isempty(stack)
        state, node = popfirst!(stack)
        statepool = Dict()
        for link in links(state)
            key = (link.dest.pdfindex, link.dest.label)
            pstate = get(statepool, key, link.dest)
            statepool[key] = pstate
            s = (link.dest.pdfindex, link.dest.label)
            weight, nextlvl = get(node, s, (-Inf, Dict()))
            node[s] = (logaddexp(link.weight, weight), nextlvl, pstate)
            push!(stack, (link.dest, node[s][2]))
        end
    end

    # 2. Build the new fsm from the tree
    nfsm = FSM{T}()
    stack = [(initstate(nfsm), tree)]
    newstates = Dict()
    newlinks = Set()
    while ! isempty(stack)
        src, node = popfirst!(stack)

        for key in keys(node)
            (pdfindex, label) = key
            weight, nextlvl, state = node[key]

            if state ∉ keys(newstates)
                if isfinal(state)
                    newstates[state] = finalstate(nfsm)
                else
                    newstates[state] = addstate!(nfsm, pdfindex = pdfindex, label = label)
                end
            end

            dest = newstates[state]
            if (src, dest) ∉ newlinks
                link!(src, dest, weight)
                push!(newlinks, (src, dest))
            end

            if ! isfinal(state)
                push!(stack, (dest, nextlvl))
            end
        end
    end

    nfsm
end

"""
    minimize(fsm)

Merge equivalent states such to reduce the size of the FSM. Only
the states that have the same `pdfindex` and the same `label` can be
potentially merged.

!!! warning
    The input FSM should not contain cycles otherwise the algorithm
    will never end.

# Examples
```julia-repl
julia> fsm = union(LinearFSM(["a", "b", "c"], Dict("a"=>1)), LinearFSM(["a", "d", "c"], Dict("a"=>1)))
julia> fsm |> minimize!
```

Input:

![See the online documentation to visualize the image](images/minimize_input.svg)

Output:

![See the online documentation to visualize the image](images/minimize_output.svg)
"""
minimize(fsm::AbstractFSM) = (weightnormalize ∘ transpose ∘ _leftminimize ∘ transpose ∘ _leftminimize ∘ _distribute)(fsm)

#######################################################################
# NIL state removal

"""
    removenilstates!(fsm)

Remove all states that are non-emitting and have no labels (except the
the initial and final states)

# Examples
```julia-repl
julia> fsm = LinearFSM(["a", "b"], Dict("a" => 1))
julia> nil = addstate!(fsm)
julia> link!(fsm, initstate(fsm), nil)
julia> link!(fsm, nil, finalstate(fsm))
julia> fsm = fsm |> weightnormalize!
julia> fsm |> removenilstates!
```
Input:

![See the online documentation to visualize the image](images/rmnil_input.svg)

Ouput:

![See the online documentation to visualize the image](images/rmnil_output.svg)

"""
function removenilstates(fsm::AbstractFSM{T}) where T
    nfsm = FSM{T}()

    newstates = Dict(initstate(fsm) => initstate(nfsm),
                     finalstate(fsm) => finalstate(nfsm))
    for state in states(fsm)
        (! islabeled(state) && ! isemitting(state)) && continue
        newstates[state] = addstate!(nfsm, pdfindex = state.pdfindex,
                                     label = state.label)
    end

    newlinks = Dict()
    stack = [(initstate(fsm), initstate(fsm), 0.0)]
    visited = Set([initstate(fsm)])
    while ! isempty(stack)
        src, state, weight = popfirst!(stack)
        for link in links(state)
            if link.dest ∈ keys(newstates)
                link!(newstates[src], newstates[link.dest], weight + link.weight)
                if link.dest ∉ visited
                    push!(visited, link.dest)
                    push!(stack, (link.dest, link.dest, 0.0))
                end
            else
                push!(stack, (src, link.dest, weight + link.weight))
            end
        end
    end

    nfsm
end

function Base.replace!(
    fsm::FSM,
    state::State,
    subfsm::FSM
)
    incoming = [link for link in parents(fsm, state)]
    outgoing = [link for link in links(fsm, state)]
    removestate!(fsm, state)
    idmap = Dict{StateID, State}()
    for s in states(subfsm)
        label = s.id == finalstateid ? "$(state.label)" : s.label
        ns = addstate!(fsm, pdfindex = s.pdfindex, label = label)
        idmap[s.id] = ns
    end

    for link in links(subfsm)
        link!(fsm, idmap[link.src.id], idmap[link.dest.id], link.weight)
    end

    for l in incoming link!(fsm, l.dest, idmap[initstateid], l.weight) end
    for l in outgoing link!(fsm, idmap[finalstateid], l.dest, l.weight) end
    fsm
end

"""
    compose!(fsm, subfsms)

Replace each state `s` in `fsm` by a "subfsms" from `subfsms` with
associated label `s.label`. `subfsms` should be a Dict{<:Label, FSM}`.

# Examples
```julia-repl
julia> fsm = union(LinearFSM(["a", "b"]), LinearFSM(["c"])) |> weightnormalize!
julia> subfsms = subfsms = Dict(
    "a" => LinearFSM(["a1", "a2", "a3"], Dict("a1"=>1, "a2"=>2, "a3"=>3)) |> addselfloop!,
    "b" => LinearFSM(["b1", "b2"], Dict("b1"=>4, "b2"=>5)) |> addselfloop!,
    "c" => LinearFSM(["c1", "c2"], Dict("c1"=>6, "c2"=>1)) |> addselfloop!
)
julia> compose!(fsm, sufsms)
```

Input :
  * `fsm`
  ![See the online documentation to visualize the image](images/compose_input1.svg)
  * `subfsms["a"]`
  ![See the online documentation to visualize the image](images/compose_input2.svg)
  * `subfsms["b"]`
  ![See the online documentation to visualize the image](images/compose_input3.svg)
  * `subfsms["c"]`
  ![See the online documentation to visualize the image](images/compose_input4.svg)
Output:
  ![See the online documentation to visualize the image](images/compose_output.svg)

Alternatively, FSMs can be composed with the `∘` operator:
```julia-replp
julia> fsm ∘ sufsms
```
When using the `∘` operator, the composition is not
performed in place.
"""
function compose(fsm::AbstractFSM{T}, subfsms::Dict) where T
    nfsm = FSM{T}()

    newsrcs = Dict(initstate(fsm) => initstate(nfsm),
                   finalstate(fsm) => finalstate(nfsm))
    newdests = Dict(initstate(fsm) => initstate(nfsm),
                    finalstate(fsm) => finalstate(nfsm))

    for state in states(fsm)
        (isinit(state) || isfinal(state)) && continue
        if state.label ∈ keys(subfsms)
            s_fsm = subfsms[state.label]

            newstates = Dict(initstate(s_fsm) => addstate!(nfsm),
                             finalstate(s_fsm) => addstate!(nfsm, label = state.label))
            for state2 in states(s_fsm)
                (isinit(state2) || isfinal(state2)) && continue
                newstates[state2] = addstate!(nfsm, pdfindex = state2.pdfindex,
                                              label = state2.label)
            end

            for link in links(s_fsm)
                link!(newstates[link.src], newstates[link.dest], link.weight)
            end

            newdests[state] = newstates[initstate(s_fsm)]
            newsrcs[state] = newstates[finalstate(s_fsm)]
        else
            nstate = addstate!(nfsm, pdfindex = state.pdfindex,
                               label = state.label)
            newsrcs[state] = nstate
            newdests[state] = nstate
        end
    end

    for link in links(fsm)
        link!(newsrcs[link.src], newdests[link.dest], link.weight)
    end

    nfsm
end

Base.:∘(subfsms::Dict, fsm::AbstractFSM) = compose(fsm, subfsms)

