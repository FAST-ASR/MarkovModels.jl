# Implementation of generic graph algorithms.

using StatsFuns

#######################################################################
# Different strategy to prune the graph during inference.

export PruningStrategy
export ThresholdPruning


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

function (pruning::ThresholdPruning)(candidates::Dict{State, T}) where T <: AbstractFloat
    maxval = maximum(p -> p.second, candidates)
    filter!(p -> maxval - p.second ≤ pruning.Δ, candidates)
end

#######################################################################
# Baum-Welch (forward-backward) algorithm

export αrecursion
export βrecursion
export αβrecursion

"""
    αrecursion(graph, llh[, pruning = ...])

Forward step of the Baum-Welch algorithm in the log-domain.
"""
function αrecursion(g::FSM, llh::Matrix{T};
                    pruning::Union{Real, NoPruning} = nopruning) where T <: AbstractFloat
    pruning! = pruning ≠ nopruning ? ThresholdPruning(pruning) : pruning

    activestates = Dict{State, T}(initstate(g) => T(0.0))
    α = Vector{Dict{State, T}}()

    for n in 1:size(llh, 2)
        push!(α, Dict{State,T}())
        for (state, weightpath) in activestates
            for (nstate, linkweight) in emittingstates(forward, state)
                nweightpath = weightpath + linkweight
                α[n][nstate] = llh[pdfindex(nstate), n] + logaddexp(get(α[n], nstate, T(-Inf)), nweightpath)
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
function βrecursion(g::FSM, llh::Matrix{T};
                    pruning::Union{Real, NoPruning} = nopruning) where T <: AbstractFloat
    pruning! = pruning ≠ nopruning ? ThresholdPruning(pruning) : pruning

    activestates = Dict{State, T}()
    β = Vector{Dict{State, T}}()
    push!(β, Dict(s => T(0.0) for (s, w) in emittingstates(backward, finalstate(g))))

    for n in size(llh, 2)-1:-1:1
        # Update the active tokens
        empty!(activestates)
        merge!(activestates, pruning!(β[1]))

        pushfirst!(β, Dict{State,T}())
        for (state, weightpath) in activestates
            prev_llh = llh[pdfindex(state), n+1]
            for (nstate, linkweight) in emittingstates(backward, state)
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
function αβrecursion(g::FSM, llh::Matrix{T};
                     pruning::Union{Real, NoPruning} = nopruning) where T <: AbstractFloat
    α = αrecursion(g, llh, pruning = pruning)
    β = βrecursion(g, llh, pruning = pruning)

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
    fs = foldl((acc, (s, w)) -> push!(acc, s), emittingstates(backward, finalstate(g)); init=[])
    ttl = filter(s -> s[1] in fs, α[end]) |> values |> sum

    γ, ttl
end

# function total_llh()

#######################################################################
# Viterbi algorithm (find the best path)

export viterbi

function maxβrecursion(g::FSM, llh::Matrix{T}, α::Vector{Dict{State,T}}) where T <: AbstractFloat
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
function viterbi(g::FSM, llh::Matrix{T};
                     pruning::Union{Real, NoPruning} = nopruning) where T <: AbstractFloat
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

#######################################################################
# Determinization algorithm

export determinize

"""
    determinize(graph)


Create a new graph where each states are connected by at most one link.
"""
function determinize(fsm::FSM)
    newfsm = FSM(fsm.emissions_names)

    newstates = Dict{StateID, State}()
    for state in states(fsm)
        newstates[state.id] = State(state.id, state.pdfindex)
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

    for state in values(newstates) addstate!(newfsm, state) end
    for arc in newarcs link!(newfsm, arc[1][1], arc[1][2], arc[2]) end

    newfsm
end

#######################################################################
# Normalization algorithm

export weightnormalize

"""
    weightnormalize(fsm)

Create a new FSM with the same topology as `fsm` such that the
sum of the exponentiated weights of the outgoing links from one state
will sum up to one.
"""
function weightnormalize(fsm::FSM)
    newfsm = FSM(fsm.emissions_names)

    totweight = Dict{StateID, Real}()
    for state in states(fsm)
        addstate!(newfsm, State(state.id, state.pdfindex))

        if state.id ∉ keys(fsm.links) continue end

        totweight[state.id] = -Inf
        for link in fsm.links[state.id]
            totweight[state.id] = logaddexp(totweight[state.id], link.weight)
        end
    end

    for link in links(fsm)
        src = newfsm.states[link.src.id]
        dest = newfsm.states[link.dest.id]
        link!(newfsm, src, dest, link.weight - totweight[link.src.id])
    end

    return newfsm

    newstates = Dict{StateID, State}()
    for state in states(g)
        newstates[id(state)] = State(id(state), pdfindex(state), name(state))
    end

    newarcs = Vector{Tuple{State, State, Real}}()
    for state in states(g)
        lognorm = reduce(logaddexp, [link.weight for link in children(state)], init=-Inf)
        for link in children(state)
            src = newstates[id(state)]
            dest = newstates[id(link.dest)]
            push!(newarcs, (src, dest, link.weight - lognorm))
        end
    end

    for state in values(newstates) addstate!(newg, state) end
    for arc in newarcs link!(arc[1], arc[2], arc[3]) end

    newg
end


#######################################################################
# Add a self-loop

export addselfloop

"""
    addselfloop(graph[, loopprob = 0.5]))

Add a self-loop to all emitting states of the graph.
"""
function addselfloop(graph::FSM; loopprob = 0.5)
    g = deepcopy(graph)
    for state in states(g)
        if isemitting(state)
            #link!(state, state, log(loopprob))
            newlinks = [(state, log(loopprob))]
            for link in children(state)
                push!(newlinks, (link.dest, link.weight + log(1 - loopprob)))
            end
            empty!(state.outgoing)
            for (dest, weight) in newlinks
                link!(state, dest, weight)
            end
        end
    end
    g
end

#######################################################################
# Union two graphs into one

import Base: union

"""
    union(fsm1::FSM, fsm2::FSM)
"""
function Base.union(fsm1::FSM, fsm2::FSM)
    fsm = FSM(merge(fsm1.emissions_names, fsm2.emissions_names))

    statecount = 0
    old2new1 = Dict{State, State}(
        initstate(fsm1) => initstate(fsm),
        finalstate(fsm1) => finalstate(fsm),
    )
    for (i, state) in enumerate(states(fsm1))
        if state.id == finalstateid || state.id == initstateid continue end
        statecount += 1
        old2new1[state] = addstate!(fsm, State(statecount, state.pdfindex))
    end

    old2new2 = Dict{State, State}(
        initstate(fsm2) => initstate(fsm),
        finalstate(fsm2) => finalstate(fsm),
    )
    for (i, state) in enumerate(states(fsm2))
        if state.id == finalstateid || state.id == initstateid continue end
        statecount += 1
        old2new2[state] = addstate!(fsm, State(statecount, state.pdfindex))
    end

    for link in links(fsm1)
        link!(fsm, old2new1[link.src], old2new1[link.dest], link.weight)
    end

    for link in links(fsm2)
        link!(fsm, old2new2[link.src], old2new2[link.dest], link.weight)
    end

    fsm |> weightnormalize
end

#######################################################################
# FSM minimization

export unreachablestates

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

export prefixes

# Compute the set of all possible pdfindex sequences "prefixing" each
# state
function prefixes(fsm::FSM, start::State, nextlinks::Function)
    retval = Dict{StateID, Set{Tuple}}(start.id => Set([()]))

    tovisit = State[start]
    while length(tovisit) > 0
        state = pop!(tovisit)

        for link in nextlinks(fsm, state)
            next = link.dest
            set = get(retval, next.id, Set{Tuple}())
            retval[next.id] = union(
                set,
                Set([(p..., state.pdfindex) for p in retval[state.id]])
            )
            push!(tovisit, next)
        end
    end

    retval
end
prefixes(fsm, ::Forward) = prefixes(fsm, initstate(fsm), children)
prefixes(fsm, ::Backward) = prefixes(fsm, finalstate(fsm), parents)

# True if two states can be merged together
function mergeable(s1, s2, prefixmap, suffixmap)
    tmp = issetequal(prefixmap[s1.id], prefixmap[s2.id])
    tmp = tmp || issetequal(suffixmap[s1.id], suffixmap[s2.id])
    tmp && (s1.pdfindex == s2.pdfindex)
end

export minimize
export distribute

# propagate the weight of each link through the graph
function distribute(fsm::FSM)
    newfsm = FSM(fsm.emissions_names)
    for state in states(fsm)
        addstate!(newfsm, State(state.id, state.pdfindex))
    end

    queue = Tuple{State, Float64}[(initstate(newfsm), 0.0)]
    while ! isempty(queue)
        state, weightpath = pop!(queue)
        for link in children(fsm, state)
            link!(newfsm, state, newfsm.states[link.dest.id], link.weight + weightpath)
            push!(queue, (link.dest, link.weight + weightpath))
        end
    end
    newfsm
end

"""
    minimize(fsm)

Return an equivalent FSM which has the minimum number of states. Only
the states that have the same `pdfindex` can be potentially merged.

Warning: `fsm` should not contain cycle !!
"""
function minimize(fsm::FSM)
    # Make sure we won't change the user's object
    fsm = deepcopy(fsm)

    # Before the "actual" minimization algorithm, we remove the
    # unreachable states to avoid some erratic behavior.
    for state in unreachablestates(fsm, forward)
        removestate!(fsm, state)
    end
    for state in unreachablestates(fsm, backward)
        removestate!(fsm, state)
    end

    # Remove the non-emitting states
    toremove = State[]
    for state in states(fsm)
        if (state.id == initstateid || state.id == finalstateid) continue end
        if ! isemitting(state)
            push!(toremove, state)
            display(state)
            for l1 in parents(fsm, state)
                for l2 in children(fsm, state)
                    link!(fsm, l1.dest, l2.dest, l1.weight + l2.weight)
                end
            end
        end
    end
    for state in toremove removestate!(fsm, state) end


    # Distribute the weights of each link through the graph to preserve
    # the proper weighting of the graph
    # I haven't thoroughly check this method so this may not be very
    # reliable
    fsm = distribute(fsm)

    prefixmap = prefixes(fsm, forward)
    suffixmap = prefixes(fsm, backward)

    sids = filter(sid -> sid ≠ initstateid &&  sid ≠ finalstateid,
                  keys(prefixmap))

    newfsm = FSM(fsm.emissions_names)
    count = 0
    smap = Dict{StateID, State}(
        initstateid => initstate(newfsm),
        finalstateid => finalstate(newfsm)
    )
    while length(sids) > 0
        sid1 = pop!(sids)
        count += 1
        smap[sid1] = addstate!(newfsm, State(count, fsm.states[sid1].pdfindex))

        toremove = Vector{StateID}()
        for sid2 in sids
            if mergeable(fsm.states[sid1], fsm.states[sid2], prefixmap, suffixmap)
                smap[sid2] = smap[sid1]
                push!(toremove, sid2)
            end
        end
        filter!(s -> s ∉ toremove, sids)
    end

    for link in links(fsm)
        link!(newfsm, smap[link.src.id], smap[link.dest.id], link.weight)
    end

    newfsm |> weightnormalize |> determinize
end

