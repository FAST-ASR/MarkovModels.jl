# Implementation of common FSM operations.

"""
    addselfloop!(fsm[, looplogprob = log(1/2)])

Add a self-loop to all emitting states of `fsm`. `looplogprob` is the
log-probability of self-transition.

# Examples
```julia-repl
julia> fsm = LinearFSM(["a", "b", "c"], Dict("a" => 1, "b" => 2))
julia> addselfloop!(fsm, log(1/4))
```
Input:

![See the online documentation to visualize the image](images/addselfloop_input.svg)

Output:

![See the online documentation to visualize the image](images/addselfloop_output.svg)

```julia-repl
julia> fsm = LinearFSM(["a", "b", "c"], Dict("a" => 1, "b" => 2))
julia> fsm |> addselfloop!
```
Input:

![See the online documentation to visualize the image](images/addselfloop_input.svg)

Output:

![See the online documentation to visualize the image](images/addselfloop_output2.svg)
"""
function addselfloop!(
    fsm::FSM,
    looplogprob::Real = log(1/2)
)
    for s in states(fsm)
        if isemitting(s)
            links = [l for l in children(fsm, s)]
            for l in links unlink!(fsm, l.src, l.dest) end
            for l in links
                link!(fsm, l.src, l.dest, l.weight + log(1 - exp(looplogprob)))
            end
            link!(fsm, s, s, looplogprob)
        end
    end
    fsm
end

"""
    determinize!(fsm)

Transform `fsm` such that each state has at most one link to any other
states.
"""
function determinize!(fsm::FSM)
    toremove = Link[]
    for s in states(fsm)
        empty!(toremove)
        dests = Dict()
        for l in children(fsm, s)
            w = get(dests, l.dest, -Inf)
            dests[l.dest] = logaddexp(w, l.weight)
        end

        for (d, w) in dests
            unlink!(fsm, s, d)
            link!(fsm, s, d)
        end
    end
    fsm
end

"""
    weightnormalize(fsm)

Create a new FSM with the same topology as `fsm` such that the
sum of the exponentiated weights of the outgoing links from one state
will sum up to one.
"""
function weightnormalize!(fsm::FSM)
    for s in states(fsm)
        total = -Inf
        dests = Dict{State, Float64}()
        total = -Inf
        for l in children(fsm, s)
            dests[l.dest] = l.weight
            total = logaddexp(total, l.weight)
        end
        for d in keys(dests) unlink!(fsm, s, d) end
        for (d, w) in dests link!(fsm, s, d, w - total) end
    end
    fsm
end

"""
    union(fsm1, fsm2, ...)

Merge several FSMs into a single one.
"""
function Base.union(
    fsm1::FSM,
    fsm2::FSM
)
    fsm = FSM()

    smap = Dict{State, State}(initstate(fsm1) => initstate(fsm),
                              finalstate(fsm1) => finalstate(fsm))
    for s in states(fsm1)
        if s.id == finalstateid || s.id == initstateid continue end
        smap[s] = addstate!(fsm, pdfindex = s.pdfindex, label = s.label)
    end
    for l in links(fsm1) link!(fsm, smap[l.src], smap[l.dest], l.weight) end

    smap = Dict{State, State}(initstate(fsm2) => initstate(fsm),
                              finalstate(fsm2) => finalstate(fsm))
    for s in states(fsm2)
        if s.id == finalstateid || s.id == initstateid continue end
        smap[s] = addstate!(fsm, pdfindex = s.pdfindex, label = s.label)
    end
    for l in links(fsm2) link!(fsm, smap[l.src], smap[l.dest], l.weight) end

    fsm
end
Base.union(fsm::FSM, rest::FSM...) = foldl(union, rest, init=fsm)

"""
    concat(fsm1, fsm2, ...)

Concatenate several FSMs into single FSM.
"""
function concat(fsm1::FSM, fsm2::FSM)
    fsm = FSM()

    cs = addstate!(fsm) # special non-emitting state for concatenaton

    smap = Dict{State, State}(initstate(fsm1) => initstate(fsm),
                              finalstate(fsm1) => cs)
    for s in states(fsm1)
        if s.id == finalstateid || s.id == initstateid continue end
        smap[s] = addstate!(fsm, pdfindex = s.pdfindex, label = s.label)
    end
    for l in links(fsm1) link!(fsm, smap[l.src], smap[l.dest], l.weight) end

    smap = Dict{State, State}(initstate(fsm2) => cs,
                              finalstate(fsm2) => finalstate(fsm))
    for s in states(fsm2)
        if s.id == finalstateid || s.id == initstateid continue end
        smap[s] = addstate!(fsm, pdfindex = s.pdfindex, label = s.label)
    end
    for l in links(fsm2) link!(fsm, smap[l.src], smap[l.dest], l.weight) end

    fsm
end
concat(fsm1::FSM, rest::FSM...) = foldl(concat, rest, init=fsm1)

"""
    removenilstates!(fsm)

Remove all states that are non-emitting and have no labels (except the
the initial and final states)
"""
function removenilstates!(fsm::FSM)
    toremove = State[]
    for state in states(fsm)
        if (state.id == initstateid || state.id == finalstateid) continue end

        # As "nil state" is a non-emitting state with no label
        if ! isemitting(state) && ! islabeled(state)
            push!(toremove, state)

            # Reconnect the states
            for l1 in parents(fsm, state)
                for l2 in children(fsm, state)
                    link!(fsm, l1.dest, l2.dest, l1.weight + l2.weight)
                end
            end
        end
    end

    for state in toremove removestate!(fsm, state) end
    fsm
end

# Returns the list of the unreachable states
function unreachablestates(
    fsm::FSM,
    start::State,
    nextlinks::Function
)
    reachable = Set{StateID}()
    tovisit = StateID[start.id]
    visited = Set{StateID}()
    while length(tovisit) > 0
        stateid = pop!(tovisit)
        push!(reachable, stateid)
        push!(visited, stateid)
        for link in nextlinks(fsm, fsm.states[stateid])
            if link.dest.id ∉ tovisit && link.dest.id ∉ visited
                push!(tovisit, link.dest.id)
            end
        end
    end
    [fsm.states[id] for id in filter(s -> s ∉ reachable, keys(fsm.states))]
end
unreachablestates(fsm::FSM, ::Forward) = unreachablestates(fsm, initstate(fsm), children)
unreachablestates(fsm::FSM, ::Backward) = unreachablestates(fsm, finalstate(fsm), parents)

# propagate the weight of each link through the graph
function distribute!(fsm::FSM)
    visited = Set{StateID}()
    queue = Tuple{State, Float64}[(initstate(fsm), 0.0)]
    while ! isempty(queue)
        state, weightpath = pop!(queue)
        push!(visited, state.id)
        links = [l for l in children(fsm, state)]
        for l in links
            unlink!(fsm, state, l.dest)
            w = l.weight + weightpath
            link!(fsm, state, l.dest, w)
            if l.dest.id ∉ visited push!(queue, (l.dest, w)) end
        end
    end
    fsm
end

"""
    minimize!(fsm)

Return an equivalent FSM which has the minimum number of states. Only
the states that have the same `pdfindex` and the same `label` can be
potentially merged.

Warning: `fsm` should not contain cycle !!
"""
function minimize!(fsm::FSM)
    # Remove states that are not reachabe from the initial/final state
    for state in unreachablestates(fsm, forward) removestate!(fsm, state) end
    for state in unreachablestates(fsm, backward) removestate!(fsm, state) end

    removenilstates!(fsm)

    # Distribute the weights of each link through the graph to preserve
    # the proper weighting of the graph
    # I haven't thoroughly check this method so this may not be very
    # reliable
    fsm = distribute!(fsm)

    # Merge states that are "equivalent"
    minimize!(fsm, forward)
    minimize!(fsm, backward)

    fsm |> weightnormalize!
end

function minimize!(
    fsm::FSM,
    s::State,
    nextlinks::Function,
    prevlinks::Function
)
    leaves = Dict()
    for l in nextlinks(fsm, s)
        if (isinit(l.dest) || isfinal(l.dest)) continue end

        context = Set([nl.dest for nl in prevlinks(fsm, l.dest)])
        key = (l.dest.pdfindex, l.dest.label, context)
        leaf, weight = get(leaves, key, (Set(), -Inf))
        push!(leaf, l.dest)
        leaves[key] = (leaf, logaddexp(weight, l.weight))
    end

    olds = State[]
    for (key, value) in leaves
        ns = addstate!(fsm, pdfindex = key[1], label = key[2])
        dests1 = Dict{State, Real}()
        dests2 = Dict{State, Real}()
        for old in value[1]
            push!(olds, old)

            for l in children(fsm, old)
                w = get(dests1, l.dest, -Inf)
                dests1[l.dest] = logaddexp(w, l.weight)
            end

            for l in parents(fsm, old)
                w = get(dests2, l.dest, -Inf)
                dests2[l.dest] = logaddexp(w, l.weight)
            end
        end
        for (d, w) in dests1 link!(fsm, ns, d, w) end
        for (d, w) in dests2 link!(fsm, d, ns, w) end
    end
    for old in olds removestate!(fsm, old) end

    for l in nextlinks(fsm, s)
        minimize!(fsm, l.dest, nextlinks, prevlinks)
    end
    fsm
end
minimize!(f::FSM, ::Forward) = minimize!(f, initstate(f), children, parents)
minimize!(f::FSM, ::Backward) = minimize!(f, finalstate(f), parents, children)

function Base.replace!(
    fsm::FSM,
    state::State,
    subfsm::FSM
)
    incoming = [link for link in parents(fsm, state)]
    outgoing = [link for link in children(fsm, state)]
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
associated label `s.label`. `subfsms` should be a Dict{Label, FSM}`.
"""
function compose!(fsm::FSM, subfsms::Dict{Label, FSM})
    toreplace = State[]
    for state in states(fsm)
        if state.label ∈ keys(subfsms) push!(toreplace, state) end
    end
    for state in toreplace replace!(fsm, state, subfsms[state.label]) end
    fsm
end

