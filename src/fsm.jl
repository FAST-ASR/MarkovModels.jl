# MarkovModels - Implementation of Finite State Machine (FSM)
#
# Lucas Ondel, 2020

#######################################################################
# Types

abstract type AbstractState end

abstract type AbstractLink{T} end

"""
    const PdfIndex = Union{UInt64, Nothing}

Type of the state pdf index.
"""
const PdfIndex = Union{UInt64, Nothing}

"""
    const Label = Union{AbstractString, Nothing}
Type of the state label.
"""
const Label = Union{AbstractString, Nothing}

"""
    InitStateID

A type with no fields whose singleton instance [`initstateid`](@ref)
is used to represent the identifier of an initial state in a graph.
See also [`FinalStateID`](@ref).
"""
struct InitStateID end

"""
    initstateid

Singleton instance of type [`InitStateID`](@ref) representing the
identifier of an initial state in a graph. See also [`finalstateid`](@ref).
"""
const initstateid = InitStateID()

"""
    FinalStateID

A type with no fields whose singleton instance [`finalstateid`](@ref)
is used to represent the identifier of a final state in a graph.
See also [`InitStateID`](@ref).
"""
struct FinalStateID end

"""

Singleton instance of type [`FinalStateID`](@ref) representing the
identifier of a final state in a graph. See also [`initstateid`](@ref).
"""
const finalstateid = FinalStateID()

"""
    const StateID = Union{Int64, InitStateID, FinalStateID}

Type of the state identifier.
"""
const StateID = Union{UInt64, InitStateID, FinalStateID}

#######################################################################
# Concrete state

"""
    struct State
        id
        pdfindex
        label
    end

State of a FSM.
  * `id` is the unique identifier of the state within a FSM.
  * `pdfindex` is the index of a probability density associated to the
     state. If the state is non-emitting, `pdfindex` is equal to
     `nothing`.
  * `label` is a readable name (either `String` or `Nothing`).
# Examples
```julia-repl
julia> State(1)
State(1)
julia> State(1, pdfindex = 2)
State(1, pdfindex = 2)
```
"""
struct State <: AbstractState
    id::StateID
    pdfindex::PdfIndex
    label::Label
    links::Vector{<:AbstractLink}
end
State(id; pdfindex = nothing, label = nothing) = State(id, pdfindex, label,
                                                       Vector{Link}())

"""
    isemitting(state)

Returns `true` if the `state` is associated with a probability density.
"""
isemitting(s::AbstractState) = ! isnothing(s.pdfindex)

"""
    isinit(state)

Returns `true` if the `state` is the initial state of the FSM.
"""
isinit(s::State) = s.id == initstateid

"""
    isfinal(state)

Returns `true` if the `state` is the final state of the FSM.
"""
isfinal(s::State) = s.id == finalstateid

"""
    islabeled(state)

Returns `true` if the `state` has a label.
"""
islabeled(s::State) = ! isnothing(s.label)

#######################################################################
# Concrete Link

"""
    struct Link{T,S,D}
        src::S
        dest::D
        weight::T
    end

Weighted link pointing from a state `src` to a state `dest` with
weight `weight`.  `T` is the type of the weight value and `S` is the
type of the source state and `D` is the type of the output state.
The weight represents the log-probability of going through this link.
"""
struct Link{T,S,D} <: AbstractLink{T}
    src::S
    dest::D
    weight::T
end

#######################################################################
# FSM

mutable struct StateIDCounter
    count::UInt64
end

struct FSM{T}
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
    fsm.idcounter.count += 1
    s = State(fsm.idcounter.count, pdfindex, label, Vector{Link}())
    fsm.states[s.id] = s
end

"""
    removestate!(fsm, state)

Remove `state` from `fsm`.
"""
#function removestate!(fsm, s)
#    # Remove all the connections of `s` before to remove it
#    toremove = State[]
#    for link in links(s) push!(toremove, link.dest) end
#    for s2 in toremove unlink!(fsm, s, s2) end
#
#    for s in states()
#    for link in parents(fsm, s) push!(toremove, link.dest) end
#    for s2 in toremove unlink!(fsm, s2, s) end
#
#    delete!(fsm.states, s.id)
#    s
#end

"""
    link!(src, dest[, weight = 0])

Add a weighted connection between `state1` and `state2`. By default,
`weight = 0`.
"""
link!(src, dest, weight = 0) = push!(src.links, Link(src, dest, weight))

"""
    unlink!(fsm, src, dest)

Remove all the connections from `src` to `dest` in `fsm`.
"""
#unlink!(fsm, src, dest) = filter!(l -> l.src ≠ src && l.dest ≠ dest, src.links)

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
# Convenience function to access particular property/attribute of the
# FSM

"""
    initstate(fsm)

Returns the initial state of `fsm`.
"""
initstate(fsm::FSM) = fsm.states[initstateid]

"""
    finalstate(fsm)

Returns the final state of `fsm`.
"""
finalstate(fsm::FSM) = fsm.states[finalstateid]

#######################################################################
# Iterators

"""
    states(fsm)

Iterator over the state of `fsm`.
"""
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

"""
    links(fsm)

Iterator over the links of the FSM.
"""
links(fsm::FSM) = LinkIterator(fsm, states(fsm))

"""
    links(state)

Iterator over the link to the children (i.e. next states) of `state`.
"""
links(state::AbstractState) = state.links

struct EmittingStatesIterator{T}
    state::T
end

function Base.iterate(iter::EmittingStatesIterator, iterstate = nothing)
    if iterstate == nothing
        T = typeof(iter.state)
        iterstate = iter.state, 1, Set(T[]), Set(T[iter.state])
    end

    curstate, idx, tovisit, visited = iterstate
    while idx <= length(curstate.links)
        dest = curstate.links[idx].dest
        if isemitting(dest) || isfinal(dest)
            return dest, (curstate, idx+1, tovisit, visited)
        elseif dest ∉ visited
            push!(tovisit, dest)
        end
        idx += 1
    end

    if isempty(tovisit) return nothing end

    curstate = pop!(tovisit)
    push!(visited, curstate)
    iterate(iter, (curstate, 1, tovisit, visited))
end

"""
    emittingstates(fsm, state, forward | backward)

Iterator over the next (forward) or previous (backward) emitting
states. For each value, the iterator return a tuple
`(nextstate, weightpath, path)`. The weight path is the sum of
the weights for all the link to reach `nextstate`. Path is a
vector of links between `state` and `nextstate`.
"""
emittingstates(s::State) = EmittingStatesIterator(s)

"""
    emittingstates(fsm)

Returns an iterator over all the emitting states of the FSM.
"""
function emittingstates(fsm::FSM)
    values(filter(p -> isemitting(p.second), fsm.states))
end

"""
    finalemittingstates(fsm)

Returns the emmiting states, which has the link with finalstate
and the corresponding paths.
"""
function finalemittingstates(fsm::FSM)
    states = Dict{State, Vector}()
    for (s,_,p) in emittingstates(fsm, finalstate(fsm), backward)
        # Need to reverse the links, cause it was created in backward way
        states[s] = map(reverse!(p)) do l Link(l.dest, l.src, l.weight) end
    end
    states
end

