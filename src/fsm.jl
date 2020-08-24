# Implementation of a Finite State Machine (FSM)

#######################################################################
# Link

# This abstract type is not necessary conceptually but it is needed
# to cope with the circular dependency between Link / State
abstract type AbstractState end

"""
    struct Link{T} where T <: AbstractFloat
        src
        dest
        weight
    end

Weighted link pointing from a state `src` to a state `dest` with weight `weight`.
`T` is the type of the weight. The weight represents the log-probability of
going through this link.
"""
struct Link{T<:AbstractFloat}
    src::AbstractState
    dest::AbstractState
    weight::T
end

#######################################################################
# State

"""
    InitStateID

A type with no fields whose singleton instance [`initstateid`](@ref)
is used to represent the identifier of an initial state in a graph.
"""
struct InitStateID end

"""
    initstateid

Singleton instance of type [`InitStateID`](@ref) representing the
identifier of an initial state in a graph.
"""
const initstateid = InitStateID()
Base.show(io::IO, id::InitStateID) = print(io, "initstateid")

"""
    FinalStateID

A type with no fields whose singleton instance [`finalstateid`](@ref)
is used to represent the identifier of a final state in a graph.
"""
struct FinalStateID end

"""
    finalstateid

Singleton instance of type [`FinalStateID`](@ref) representing the
identifier of a final state in a graph.
"""
const finalstateid = FinalStateID()
Base.show(io::IO, id::FinalStateID) = print(io, "finalstateid")

"""
    const StateID = Union{Int64, InitStateID, FinalStateID}

Type of the state identifier.
"""
const StateID = Union{Int64, InitStateID, FinalStateID}

"""
    const Label = Union{AbstractString, Nothing}

Type of the state label.
"""
const Label = Union{AbstractString, Nothing}

"""
    const Pdfindex = Union{Int64, Nothing}

Type of the state pdf index.
"""
const Pdfindex = Union{Int64, Nothing}

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
    pdfindex::Union{Int64, Nothing}
    label::Union{AbstractString, Nothing}
end

function Base.show(
    io::IO,
    s::State
)
    str = "State($(s.id)"
    if ! isnothing(s.pdfindex) str = "$str, pdfindex = $(s.pdfindex)" end
    if ! isnothing(s.label) str = "$str, label = $(s.label)" end
    print(io, "$str)")
end

State(id; pdfindex = nothing, label = nothing) = State(id, pdfindex, label)

"""
    isemitting(state)

Returns `true` if the `state` is associated with a probability density.
"""
isemitting(s::State) = ! isnothing(s.pdfindex)

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
# FSM

mutable struct StateIDCounter
    count::Int64
end

struct FSM
    idcounter::StateIDCounter
    states::Dict{StateID, State}
    links::Dict{StateID, Vector{Link}}
    backwardlinks::Dict{StateID, Vector{Link}}

    FSM() = new(
        StateIDCounter(0),
        Dict{StateID, State}(
            initstateid => State(initstateid),
            finalstateid => State(finalstateid)
        ),
        Dict{StateID, Vector{Link}}(),
        Dict{StateID, Vector{Link}}(),
    )
end

#######################################################################
# Methods to construct the FSM

"""
    addstate!(fsm[, pdfindex = ..., label = "..."])

Add `state` to `fsm` and return it.
"""
function addstate!(
    fsm::FSM;
    id = nothing,
    pdfindex = nothing,
    label = nothing
)
    fsm.idcounter.count += 1
    s = State(fsm.idcounter.count, pdfindex, label)
    fsm.states[s.id] = s
end

"""
    removestate!(fsm, state)

Remove `state` from `fsm`.
"""
function removestate!(
    fsm::FSM,
    s::State
)
    # Remove all the connections of `s` before to remove it
    toremove = State[]
    for link in children(fsm, s) push!(toremove, link.dest) end
    for s2 in toremove unlink!(fsm, s, s2) end

    for link in parents(fsm, s) push!(toremove, link.dest) end
    for s2 in toremove unlink!(fsm, s2, s) end

    delete!(fsm.states, s.id)
    delete!(fsm.links, s.id)
    delete!(fsm.backwardlinks, s.id)

    s
end

"""
    link!(state1, state2[, weight])

Add a weighted connection between `state1` and `state2`. By default,
`weight = 0`.
"""
function link!(
    fsm::FSM,
    s1::State,
    s2::State,
    weight::Real = 0.
)
    array = get(fsm.links, s1.id, Vector{Link}())
    push!(array, Link(s1, s2, weight))
    fsm.links[s1.id] = array

    array = get(fsm.backwardlinks, s2.id, Vector{Link}())
    push!(array, Link(s2, s1, weight))
    fsm.backwardlinks[s2.id] = array
end

"""
    unlink!(fsm, src, dest)

Remove all the connections from `src` to `dest` in `fsm`.
"""
function unlink!(
    fsm::FSM,
    s1::State,
    s2::State
)
    if s1.id ∈ keys(fsm.links) filter!(l -> l.dest.id ≠ s2.id, fsm.links[s1.id]) end
    if s2.id ∈ keys(fsm.backwardlinks) filter!(l -> l.dest.id ≠ s1.id, fsm.backwardlinks[s2.id]) end

    nothing
end

"""
    LinearFSM(seq[, emissionsmap::Dict{<:Label, <:Pdfindex}])

Create a linear FSM from a sequence of labels `seq`. If
`emissionsmap` is provided, every item `l` of `seq` with a matching entry
in `emissionsmap` will be assigned the pdf index `emissionsmap[l]`.
"""
function LinearFSM(
    sequence::AbstractArray{<:Label},
    emissionsmap::Dict{<:Label, <:Pdfindex} = Dict{Label, Pdfindex}()
)
    fsm = FSM()
    prevstate = initstate(fsm)
    for token in sequence
        pdfindex = get(emissionsmap, token, nothing)
        s = addstate!(fsm, pdfindex = pdfindex, label = token)
        link!(fsm, prevstate, s)
        prevstate = s
    end
    link!(fsm, prevstate, finalstate(fsm))
    fsm
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

function Base.iterate(
    iter::LinkIterator,
    iterstate = nothing
)
    # Initialize the state of the iterator.
    if iterstate == nothing
        state, siterstate = iterate(iter.siter)
        liter = get(iter.fsm.links, state.id, Vector{Link}())
        next = iterate(liter)
    else
        state, siterstate, liter, literstate = iterstate
        next = iterate(liter, literstate)
    end

    while next == nothing
        nextstate = iterate(iter.siter, siterstate)

        # Finished iterating over the states.
        # End the iterations.
        if nextstate == nothing return nothing end

        state, siterstate = nextstate
        liter = get(iter.fsm.links, state.id, Vector{Link}())
        next = iterate(liter)
    end

    link, literstate = next
    newliterstate = (state, siterstate, liter, literstate)

    return link, newliterstate
end

"""
    links(fsm)

Iterator over the links of the FSM.
"""
links(fsm::FSM) = LinkIterator(fsm, states(fsm))

"""
    children(fsm, state)

Iterator over the link to the children (i.e. next states) of `state`.
"""
children(fsm::FSM, state::State) = get(fsm.links, state.id, Vector{Link}())

"""
    parents(fsm, state)

Iterator over the link to the parents (i.e. previous states) of `state`.
"""
parents(fsm::FSM, state::State) = get(fsm.backwardlinks, state.id, Vector{Link}())

struct Forward end
const forward = Forward()

struct Backward end
const backward = Backward()

struct EmittingStatesIterator
    state::State
    getlinks::Function
end

function Base.iterate(
    iter::EmittingStatesIterator,
    queue = nothing
)
    if queue == nothing
        queue = Vector([([link], oftype(link.weight, 0.0))
                for link in iter.getlinks(iter.state)])
    end

    hasfoundstate = false
    nextstate = nothing
    weight = nothing
    path = nothing
    while hasfoundstate ≠ true
        # We didn't find any emitting state, return `nothing`
        if isempty(queue) return nothing end

        # Explore the next link in the queue
        path, pathweight = pop!(queue)
        link = last(path)

        if isemitting(link.dest)
            nextstate, weight = link.dest, pathweight + link.weight
            hasfoundstate = true
        else
            append!(queue, [([path; l], pathweight + link.weight)
                            for l in iter.getlinks(link.dest)])
        end
    end
    (nextstate, weight, path), queue
end

"""
    emittingstates(fsm, state, forward | backward)

Iterator over the next (forward) or previous (backward) emitting
states. For each value, the iterator return a tuple
`(nextstate, weightpath, path)`. The weight path is the sum of 
the weights for all the link to reach `nextstate`. Path is a 
vector of links between `state` and `nextstate`.
"""
function emittingstates(
    fsm::FSM,
    s::State,
    ::Forward
)
    EmittingStatesIterator(s, st -> children(fsm, st))
end

function emittingstates(
    fsm::FSM,
    s::State,
    ::Backward
)
    EmittingStatesIterator(s, st -> parents(fsm, st))
end

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
    for (s,_,p) in emittingstates2(fsm, finalstate(fsm), backward)
        # Need to reverse the links, cause it was created in backward way
        states[s] = map(reverse!(p)) do l Link(l.dest, l.src, l.weight) end 
    end
    states
end

