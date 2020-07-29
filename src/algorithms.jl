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
function determinize(g::FSM)
    newg = FSM()

    newstates = Dict{StateID, State}()
    for state in states(g)
        newstates[id(state)] = State(id(state), pdfindex(state), name(state))
    end

    newarcs = Dict{Tuple{State, State}, Real}()
    for state in states(g)
        for link in children(state)
            src = newstates[id(state)]
            dest = newstates[id(link.dest)]
            destweight = get(newarcs, (src, dest), oftype(link.weight, -Inf))
            newweight = logaddexp(destweight, link.weight)
            newarcs[(src, dest)] = newweight
        end
    end

    for state in values(newstates) addstate!(newg, state) end
    for arc in newarcs link!(arc[1][1], arc[1][2], arc[2]) end

    newg
end

#######################################################################
# Normalization algorithm

export weightnormalize

"""
    weightnormalize(graph)

Update the weights of the graph such that the exponentiation of the
weight of all the outoing arc from a state sum up to one.
"""
function weightnormalize(g::FSM)
    newg = FSM()

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

    fsm
end

#######################################################################
# FSM minimization

export minimize

function leftminimize!(g::FSM, state::AbstractState)
    leaves = Dict()
    for link in children(state)
        leaf, weight = get(leaves, pdfindex(link.dest), ([], -Inf))
        push!(leaf, link.dest)
        leaves[pdfindex(link.dest)] = (leaf, logaddexp(weight, link.weight))
    end

    empty!(state.outgoing)
    for (nextstates, weight) in values(leaves)
        mergedstate = nextstates[1]
        filter!(l -> l.dest ≠ state, mergedstate.incoming)

        # Now we removed all the extra states of the graph.
        links = vcat([next.outgoing for next in nextstates[2:end]]...)
        for link in links
            link!(mergedstate, link.dest, link.weight)
        end

        for old in nextstates[2:end]
            for link in children(old)
                filter!(l -> l.dest ≠ old, link.dest.incoming)
            end
            delete!(g.states, id(old))
        end

        # Reconnect the previous state with the merged state
        link!(state, mergedstate, weight)

        # Minimize the subgraph.
        leftminimize!(g, mergedstate)
    end
    g
end

function rightminimize!(g::FSM, state::AbstractState)
    leaves = Dict()
    for link in parents(state)
        leaf, weight = get(leaves, pdfindex(link.dest), ([], -Inf))
        push!(leaf, link.dest)
        leaves[pdfindex(link.dest)] = (leaf, logaddexp(weight, link.weight))
    end

    empty!(state.incoming)
    for (nextstates, weight) in values(leaves)
        mergedstate = nextstates[1]
        filter!(l -> l.dest ≠ state, mergedstate.outgoing)

        # Now we removed all the extra states of the graph.
        links = vcat([next.incoming for next in nextstates[2:end]]...)
        for link in links
            #link!(mergedstate, link.dest, link.weight)
            link!(link.dest, mergedstate, link.weight)
        end

        for old in nextstates[2:end]
            for link in parents(old)
                filter!(l -> l.dest ≠ old, link.dest.outgoing)
            end
            delete!(g.states, id(old))
        end

        # Reconnect the previous state with the merged state
        link!(mergedstate, state, weight)

        # Minimize the subgraph.
        rightminimize!(g, mergedstate)
    end
    g
end

"""
    minimize(g::FSM)
"""
minimize(fsm::FSM) = begin
    newfsm = deepcopy(g)
    newg = leftminimize!(newg, initstate(newg))
    rightminimize!(newg, finalstate(newg))
end

