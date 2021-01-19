# MarkovModels - Implementation of Finite State Machine (FSM)
#
# Lucas Ondel, 2020

#######################################################################
# AbstractFSM interface

abstract type AbstractFSM{T} end

"""
    initstate(fsm)

Returns the initial state of `fsm`.
"""
initstate

"""
    finalstate(fsm)

Returns the final state of `fsm`.
"""
finalstate

"""
    states(fsm)

Iterator over the state of `fsm`.
"""
states

"""
    links(fsm)

Iterator over the links of the FSM.
"""
links(fsm::AbstractFSM)

"""
    emittingstates(fsm)

Returns an iterator over all the emitting states of the FSM.
"""
emittingstates(fsm::AbstractFSM)

#######################################################################
# FSM

mutable struct StateIDCounter
    count::UInt64
end

struct FSM{T} <: AbstractFSM{T}
    idcounter::StateIDCounter
    states::Dict{StateID, State}

    FSM{T}() where T = new{T}(
        StateIDCounter(0),
        Dict{StateID, State}(
            initstateid => State(initstateid),
            finalstateid => State(finalstateid)
        ),
    )
end

#######################################################################
# Methods to construct the FSM

"""
    addstate!(fsm[, pdfindex = ..., label = "..."])

Add `state` to `fsm` and return it.
"""
function addstate!(fsm; id = nothing, pdfindex = nothing, label = nothing)
    if isnothing(id)
        fsm.idcounter.count += 1
        id = fsm.idcounter.count
    else
        fsm.idcounter.count = max(fsm.idcounter.count, id)
    end
    s = State(id, pdfindex, label, Vector{Link}())
    fsm.states[s.id] = s
end

"""
    link!(src, dest[, weight = 0])

Add a weighted connection between `state1` and `state2`. By default,
`weight = 0`.
"""
link!(src, dest, weight = 0) = push!(src.links, Link(src, dest, weight))

"""
    LinearFSM(seq[, emissionsmap::Dict{<:Label, <:PdfIndex}])

Create a linear FSM from a sequence of labels `seq`. If
`emissionsmap` is provided, every item `l` of `seq` with a matching entry
in `emissionsmap` will be assigned the pdf index `emissionsmap[l]`.
"""
function LinearFSM(T, sequence::AbstractVector,
                   emissionsmap = Dict{Label, PdfIndex}())
    fsm = FSM{T}()
    prevstate = initstate(fsm)
    for token in sequence
        pdfindex = get(emissionsmap, token, nothing)
        s = addstate!(fsm, pdfindex = pdfindex, label = token)
        link!(prevstate, s)
        prevstate = s
    end
    link!(prevstate, finalstate(fsm))
    fsm
end
function LinearFSM(sequence::AbstractVector,
                   emissionsmap = Dict{Label, PdfIndex}())
    LinearFSM(Float64, sequence, emissionsmap)
end

#######################################################################
# Implementation of the AbstractFSM interface

initstate(fsm::FSM) = fsm.states[initstateid]
finalstate(fsm::FSM) = fsm.states[finalstateid]

states(fsm::FSM) = values(fsm.states)

struct LinkIterator
    fsm::FSM
    siter
end

function Base.iterate(iter::LinkIterator, iterstate = nothing)
    if iterstate == nothing
        init = initstate(iter.fsm)
        iterstate = init, 1, Set{typeof(init)}(), Set([init])
    end
    curstate, idx, tovisit, visited = iterstate

    if idx <= length(curstate.links)
        link = curstate.links[idx]
        push!(tovisit, link.dest)
        return link, (curstate, idx+1, tovisit, visited)
    end

    while curstate ∈ visited
        if isempty(tovisit)
            return nothing
        end
        curstate = pop!(tovisit)
    end

    push!(visited, curstate)
    iterate(iter, (curstate, 1, tovisit, visited))
end

links(fsm::FSM) = LinkIterator(fsm, states(fsm))

function emittingstates(fsm::FSM)
    values(filter(p -> isemitting(p.second), fsm.states))
end

#function finalemittingstates(fsm::FSM)
#    states = Dict{State, Vector}()
#    for (s,_,p) in emittingstates(fsm, finalstate(fsm), backward)
#        # Need to reverse the links, cause it was created in backward way
#        states[s] = map(reverse!(p)) do l Link(l.dest, l.src, l.weight) end
#    end
#    states
#end

