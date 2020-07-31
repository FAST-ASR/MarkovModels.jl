# Implementation of generic graph algorithms.

using StatsFuns: logaddexp, logsumexp
import Base: union

#######################################################################
# Different strategy to prune the graph during inference.

abstract type PruningStrategy end

# No pruning strategy for the αβ recursion
struct NoPruning <: PruningStrategy end
const nopruning = NoPruning()
(::NoPruning)(candidates::Dict{State, T}) where T <: AbstractFloat = candidates

# Pruning strategy which retain all the hypotheses highger than a
# given threshold
struct ThresholdPruning <: PruningStrategy
    Δ::Real
end

function (pruning::ThresholdPruning)(
    candidates::Dict{State, T}
) where T <: AbstractFloat
    maxval = maximum(p -> p.second, candidates)
    filter!(p -> maxval - p.second ≤ pruning.Δ, candidates)
end

#######################################################################
# Baum-Welch (forward-backward) algorithm

"""
    αrecursion(graph, llh[, pruning = ...])

Forward step of the Baum-Welch algorithm in the log-domain.
"""
function αrecursion(
    fsm::FSM,
    llh::Matrix{T};
    pruning::Union{Real, NoPruning} = nopruning
) where T <: AbstractFloat

    pruning! = pruning ≠ nopruning ? ThresholdPruning(pruning) : pruning

    activestates = Dict{State, T}(initstate(fsm) => T(0.0))
    α = Vector{Dict{State, T}}()

    for n in 1:size(llh, 2)
        push!(α, Dict{State,T}())
        for (state, weightpath) in activestates
            for (nstate, linkweight) in emittingstates(fsm, state, forward)
                nweightpath = weightpath + linkweight
                α[n][nstate] = llh[nstate.pdfindex, n] + logaddexp(get(α[n], nstate, T(-Inf)), nweightpath)
            end
        end

        empty!(activestates)
        merge!(activestates, pruning!(α[n]))
    end
    α
end

"""
    βrecursion(graph, llh[, pruning = ...])

Backward step of the Baum-Welch algorithm in the log domain.
"""
function βrecursion(
    fsm::FSM,
    llh::Matrix{T};
    pruning::Union{Real, NoPruning} = nopruning
) where T <: AbstractFloat

    pruning! = pruning ≠ nopruning ? ThresholdPruning(pruning) : pruning

    activestates = Dict{State, T}()
    β = Vector{Dict{State, T}}()
    push!(β, Dict(s => T(0.0)
         for (s, w) in emittingstates(fsm, finalstate(fsm), backward)))

    for n in size(llh, 2)-1:-1:1
        # Update the active tokens
        empty!(activestates)
        merge!(activestates, pruning!(β[1]))

        pushfirst!(β, Dict{State,T}())
        for (state, weightpath) in activestates
            prev_llh = llh[state.pdfindex, n+1]
            for (nstate, linkweight) in emittingstates(fsm, state, backward)
                nweightpath = weightpath + linkweight + prev_llh
                β[1][nstate] = logaddexp(get(β[1], nstate, T(-Inf)), nweightpath)
            end
        end
    end
    β
end

"""
    αβrecursion(graph, llh[, pruning = ...])

Baum-Welch algorithm in  the log domain.
"""
function αβrecursion(
    fsm::FSM, llh::Matrix{T};
    pruning::Union{Real, NoPruning} = nopruning
) where T <: AbstractFloat

    α = αrecursion(fsm, llh, pruning = pruning)
    β = βrecursion(fsm, llh, pruning = pruning)

    γ = Vector{Dict{State,T}}()

    for n in 1:size(llh, 2)
        push!(γ, Dict{State, T}())
        for s in union(keys(α[n]), keys(β[n]))
            a = get(α[n], s, T(-Inf))
            b = get(β[n], s, T(-Inf))
            γ[n][s] = a + b
        end
        filter!(p -> isfinite(p.second), γ[n])
        sum = logsumexp(values(γ[n]))

        for s in keys(γ[n])
            γ[n][s] -= sum
        end
    end

    # Total Log Likelihood
    fs = foldl((acc, (s, w)) -> push!(acc, s), emittingstates(fsm, finalstate(fsm), backward); init=[])
    ttl = filter(s -> s[1] in fs, α[end]) |> values |> sum

    γ, ttl
end

function maxβrecursion(
    g::FSM,
    llh::Matrix{T},
    α::Vector{Dict{State,T}}
) where T <: AbstractFloat

    bestseq = Vector{State}()
    activestates = Dict{State, T}(finalstate(g) => T(0.0))
    newstates = Dict{State, T}()

    for n in size(llh, 2):-1:1
        for (state, weightpath) in activestates
            emitting = isemitting(state)
            prev_llh = emitting ? llh[state.pdfindex, n+1] : T(0.0)
            for (nstate, linkweight) in emittingstates(g, state, backward)
                nweightpath = weightpath + linkweight + prev_llh
                newstates[nstate] = logaddexp(get(newstates, nstate, T(-Inf)), nweightpath)
            end
        end

        hypscores = Vector{T}(undef, length(newstates))
        hypstates = Vector{State}(undef, length(newstates))
        for (i, (nstate, nweightpath)) in enumerate(newstates)
            hypscores[i] = get(α[n], nstate, T(-Inf)) + nweightpath
            hypstates[i] = nstate
        end
        println(hypstates)
        println(hypscores)
        println("-----")
        maxval, maxidx = findmax(hypscores)
        best = hypstates[maxidx]
        pushfirst!(bestseq, best)

        empty!(activestates)
        activestates[best] = newstates[best]
        empty!(newstates)
    end
    bestseq
end

"""
    viterbi(graph, llh[, pruning = ...])

Viterbi algorithm.
"""
function viterbi(
    fsm::FSM,
    llh::Matrix{T};
    pruning::Union{Real, NoPruning} = nopruning
) where T <: AbstractFloat

    α = αrecursion(fsm, llh, pruning = pruning)
    path = maxβrecursion(fsm, llh, α)

    # Return the best seq as a new graph
    ng = FSM()
    prevstate = initstate(ng)
    for state in path
        s = addstate!(ng, pdfindex = state.pdfindex, label = state.label)
        link!(ng, prevstate, s)
        prevstate = s
    end
    link!(ng, prevstate, finalstate(ng), 0.)
    ng
end

"""
    addselfloop!(fsm[, loopprob = 0.5])

Add a self-loop to all emitting states.
"""
function addselfloop!(
    fsm::FSM,
    loopprob::Real = 0.5
)
    for s in states(fsm)
        if isemitting(s)
            for l in children(fsm, s) l.weight += log(1 - 0.5) end
            link!(fsm, s, s, log(loopprob))
        end
    end
    fsm
end

"""
    determinize(graph)

Create a new graph where each states are connected by at most one link.
"""
function determinize!(
    fsm::FSM,
    s::State,
    nextlinks::Function,
    visited::Vector{State}
)
    leaves = Dict()
    for l in nextlinks(fsm, s)
        if (l.dest.id == initstateid || l.dest.id == finalstateid) continue end
        if l.dest ∈  visited continue end
        leaf, weight = get(leaves, (l.dest.pdfindex, l.dest.label), (Set(), -Inf))
        push!(leaf, l.dest)
        key = (l.dest.pdfindex, l.dest.label)
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

    push!(visited, s)

    for l in nextlinks(fsm, s)
        if l.dest ∉ visited determinize!(fsm, l.dest, nextlinks, visited) end
    end
    fsm
end
determinize!(f::FSM, ::Forward) = determinize!(f, initstate(f), children, State[])
determinize!(f::FSM, ::Backward) = determinize!(f, finalstate(f), parents, State[])
determinize!(f::FSM) = determinize!(f, initstate(f), children, State[])

"""
    weightnormalize(fsm)

Create a new FSM with the same topology as `fsm` such that the
sum of the exponentiated weights of the outgoing links from one state
will sum up to one.
"""
function weightnormalize!(fsm::FSM)
    for s in states(fsm)
        total = -Inf
        for l in children(fsm, s) total = logaddexp(total, l.weight) end
        for l in children(fsm, s) l.weight -= total end
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
Base.union(fsm1::FSM, fsm2::FSM, x::Vararg{FSM}) = union(union(fsm1, fsm2), x...)
Base.union(fsm::FSM) = fsm

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
        for l in children(fsm, state)
            l.weight += weightpath
            if l.dest.id ∉ visited push!(queue, (l.dest, l.weight)) end
        end
    end
    fsm
end

function minimizestep!(fsm::FSM, state::State, nextlinks::Function)
    leaves = Dict()
    for link in nextlinks(fsm, state)
        if (link.dest.id == initstateid || link.dest.id == finalstateid) continue end

        leaf, weight = get(leaves, (link.dest.pdfindex, link.dest.label),
                           (Set(), -Inf))
        push!(leaf, link.dest)
        key = (link.dest.pdfindex, link.dest.label)
        leaves[key] = (leaf, logaddexp(weight, link.weight))
    end

    # OPTIMIZATION: we recreate all the states generating
    # lot of memory operations. We ould simply remove states...
    newstates = State[]
    for (key, value) in leaves
        s = addstate!(fsm, pdfindex = key[1], label = key[2])
        for oldstate in value[1]
            for link in children(fsm, oldstate)
                link!(fsm, s, link.dest, link.weight)
            end

            for link in parents(fsm, oldstate)
                link!(fsm, link.dest, s, link.weight)
            end
        end
        push!(newstates, s)
    end

    for (oldstates, _) in values(leaves)
        for s in oldstates
            removestate!(fsm, s)
        end
    end

    for s in newstates
        minimizestep!(fsm, s, nextlinks)
    end
end
minimizestep!(f::FSM, s::State, ::Forward) = minimizestep!(f, s, children)
minimizestep!(f::FSM, s::State, ::Backward) = minimizestep!(f, s, parents)

"""
    minimize!(fsm)

Return an equivalent FSM which has the minimum number of states. Only
the states that have the same `pdfindex` can be potentially merged.

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
    determinize!(fsm, forward)
    determinize!(fsm, backward)

    fsm |> weightnormalize!
end

function replace!(
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

