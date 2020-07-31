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
            for (nstate, linkweight) in emittingstates(backward, state)
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

    α = αrecursion(g, llh, pruning = pruning)
    path = maxβrecursion(g, llh, α)

    # Return the best seq as a new graph
    ng = FSM()
    prevstate = initstate(ng)
    for (i, state) in enumerate(path)
        s = addstate!(ng, State(i, pdfindex(state), name(state)))
        link!(prevstate, s, 0.)
        prevstate = s
    end
    link!(prevstate, finalstate(ng), 0.)
    ng
end

"""
    determinize(graph)

Create a new graph where each states are connected by at most one link.
"""
function determinize(fsm::FSM)
    newfsm = FSM()

    newstates = Dict{StateID, State}()
    for state in states(fsm)
        if state.id == initstateid
            newstates[state.id] = initstate(newfsm)
        elseif state.id == finalstateid
            newstates[state.id] = finalstate(newfsm)
        else
            newstates[state.id] = addstate!(newfsm, pdfindex = state.pdfindex,
                                            label = state.label)
        end
    end

    newarcs = Dict{Tuple{State, State}, Real}()
    for state in states(fsm)
        for link in children(fsm, state)
            src = newstates[state.id]
            dest = newstates[link.dest.id]
            destweight = get(newarcs, (src, dest), oftype(link.weight, -Inf))
            newweight = logaddexp(destweight, link.weight)
            newarcs[(src, dest)] = newweight
        end
    end

    for arc in newarcs link!(newfsm, arc[1][1], arc[1][2], arc[2]) end

    newfsm
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
        for l in children(fsm, s) total = logaddexp(total, l.weight) end
        for l in children(fsm, s) l.weight -= total end
    end
    fsm
end

"""
    addselfloop!(graph[, loopprob = 0.5]))

Add a self-loop to all emitting states of the graph.
"""
function addselfloop!(
    fsm::FSM;
    loopprob::Real = 0.5
)
    for state in states(fsm)
        if isemitting(state)
            for l in children(fsm, state) l.weight += log(1 - loopprob) end
            link!(fsm, state, state, weight)
        end
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

    old2new1 = Dict{State, State}(
        initstate(fsm1) => initstate(fsm),
        finalstate(fsm1) => finalstate(fsm),
    )
    for state in states(fsm1)
        if state.id == finalstateid || state.id == initstateid continue end
        old2new1[state] = addstate!(fsm, pdfindex = state.pdfindex,
                                    label =state.label)
    end

    old2new2 = Dict{State, State}(
        initstate(fsm2) => initstate(fsm),
        finalstate(fsm2) => finalstate(fsm),
    )
    for state in states(fsm2)
        if state.id == finalstateid || state.id == initstateid continue end
        old2new2[state] = addstate!(fsm, pdfindex = state.pdfindex,
                                    label = state.label)
    end

    for link in links(fsm1)
        link!(fsm, old2new1[link.src], old2new1[link.dest], link.weight)
    end

    for link in links(fsm2)
        link!(fsm, old2new2[link.src], old2new2[link.dest], link.weight)
    end

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
function unreachablestates(fsm::FSM, start::State, nextlinks::Function)
    reachable = Set{StateID}()
    tovisit = StateID[start.id]
    while length(tovisit) > 0
        stateid = pop!(tovisit)
        push!(reachable, stateid)
        for link in nextlinks(fsm, fsm.states[stateid])
            if link.dest.id ∉ tovisit
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
    queue = Tuple{State, Float64}[(initstate(fsm), 0.0)]
    while ! isempty(queue)
        state, weightpath = pop!(queue)
        for link in children(fsm, state)
            link.weight += weightpath
            push!(queue, (link.dest, link.weight))
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
    minimizestep!(fsm, initstate(fsm), forward)
    minimizestep!(fsm, finalstate(fsm), backward)

    fsm |> weightnormalize |> determinize
end

