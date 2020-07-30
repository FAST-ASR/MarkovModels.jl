# Implementation of a Finite State Machine (FSM)

#######################################################################
# Link

# This abstract type is not necessary conceptually but it is needed
# to cope with the circular dependency between Link / State
abstract type AbstractState end

"""
    struct Link{T} where T <: AbstractFloat
        dest
        weight
        label
    end

Weighted link pointing to a state `dest` with label `label`. `T` is
the type of the weight. The weight represents the log-probability of
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
    stuct State
        id
        pdfindex
        outgoing
        incoming
    end

State of a FSM.
  * `id` is the unique identifier of the state within a FSM.
  * `pdfindex` is the index of a probability density associated to the
     state. If the state is non-emitting, `pdfindex` is equal to
     `nothing`.
  * `outgoing` is a `Vector` of links leaving the state.
  * `incoming` is a `Vector` of links arriving to the state.

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
end

function Base.show(io::IO, s::State)
    if ! isnothing(s.pdfindex)
        print(io, "State($(s.id), pdfindex = $(s.pdfindex))")
    else
        print(io, "State($(s.id))")
    end
end

State(id; pdfindex = nothing) = State(id, pdfindex)

"""
    isemitting(state)

Returns `true` if the `state` is associated with a probability density.
"""
isemitting(s::State) = ! isnothing(s.pdfindex)

#######################################################################
# FSM

struct FSM
    states::Dict{StateID, State}
    links::Dict{StateID, Vector{Link}}
    backwardlinks::Dict{StateID, Vector{Link}}
    emissions_names::Dict{StateID, AbstractString}

    FSM(emissions_names::Dict{StateID, AbstractString}) = new(
        Dict{StateID, State}(
            initstateid => State(initstateid),
            finalstateid => State(finalstateid)
        ),
        Dict{StateID, Vector{Link}}(),
        Dict{StateID, Vector{Link}}(),
        emissions_names
    )
end
FSM() = FSM(Dict{StateID, AbstractString}())

#######################################################################
# Methods to construct the FSM

"""
    addstate!(fsm, state)

Add `state` to `fsm`.
"""
addstate!(fsm::FSM, s::State) = fsm.states[s.id] = s

"""
    removestate!(fsm, state)

Remove `state` from `fsm`.
"""
function removestate!(fsm::FSM, s::State)
    delete!(fsm.states, s.id)
    delete!(fsm.links, s.id)
    delete!(fsm.backwardlinks, s.id)
    for state in states(fsm)
        toremove = Vector{Link}()
        for link in children(fsm, state)
            if link.dest.id == s.id
                push!(toremove, link)
            end
        end
        if length(toremove) > 0
            filter!(v -> v ∉ toremove, fsm.links[state.id])
        end

        toremove = Vector{Link}()
        for link in parents(fsm, state)
            if link.dest.id == s.id
                push!(toremove, link)
            end
        end
        if length(toremove) > 0
            filter!(v -> v ∉ toremove, fsm.backwardlinks[state.id])
        end
    end
    s
end

"""
    link!(state1, state2[, weight])

Add a weighted connection between `state1` and `state2`. By default,
`weight = 0`.
"""
function link!(fsm, s1::State, s2::State, weight::Real = 0.)
    array = get(fsm.links, s1.id, Vector{Link}())
    push!(array, Link(s1, s2, weight))
    fsm.links[s1.id] = array

    array = get(fsm.backwardlinks, s2.id, Vector{Link}())
    push!(array, Link(s2, s1, weight))
    fsm.backwardlinks[s2.id] = array
end

"""
    unlink!(fsm, src, dest)

Remove all the connections betwee `src` and `dest` in `fsm`.
"""
function unlink!(fsm::FSM, src::State, dest::State)
    filter!(l -> l.dest.id ≠ dest.id, fsm.links[src.id])
    filter!(l -> l.dest.id ≠ src.id, fsm.backwardlinks[dest.id])
    nothing
end

"""
    LinearFSM(seq, emissions_names)

Create a linear FSM from a sequence of label `seq`. `emissions_names`
should be a one-to-one mapping pdfindex -> label.
"""
function LinearFSM(sequence::AbstractArray{String},
                   emissions_names::Dict{StateID, AbstractString})

    # Reverse the mapping to get the pdfindex from the labels
    rmap = Dict(v => k for (k, v) in emissions_names)

    fsm = FSM(emissions_names)
    prevstate = initstate(fsm)
    for (i, token) in enumerate(sequence)
        s = addstate!(fsm, State(i, rmap[token]))
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
    name(fsm, state)

Return the name of `state`. If the state has no name, it returns
`state.id` as a string.
"""
name(fsm, state) = get(fsm.emissions_names, state.pdfindex, "$(state.id)")

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

function Base.iterate(iter::EmittingStatesIterator, queue = nothing)
    if queue == nothing
        queue = Vector([(link, oftype(link.weight, 0.0))
                        for link in iter.getlinks(iter.state)])
    end

    hasfoundstate = false
    nextstate = weight = nothing
    while hasfoundstate ≠ true
        if isempty(queue) return nothing end
        s_w = nextemittingstates!(queue, iter.getlinks)
        if s_w ≠ nothing
            hasfoundstate = true
            nextstate, weight = s_w
        end
    end
    (nextstate, weight), queue
end

function nextemittingstates!(queue::Vector{Tuple{Link{T}, T}}, getlinks::Function) where T <: AbstractFloat
    link, pathweight = pop!(queue)
    if isemitting(link.dest)
        return link.dest, pathweight + link.weight
    end
    append!(queue, [(newlink, pathweight + link.weight) for newlink in getlinks(link.dest)])
    return nothing
end

"""
    emittingstates(fsm, state, forward | backward)

Iterator over the next (forward) or previous (backward) emitting
states. For each value, the iterator return a tuple
`(nextstate, weightpath)`. The weight path is the sum of the weights
for all the link to reach `nextstate`.
"""
function emittingstates(fsm::FSM, s::State, ::Forward)
    EmittingStatesIterator(s, st -> children(fsm, st))
end

function emittingstates(fsm::FSM, s::State, ::Backward)
    EmittingStatesIterator(s, st -> parents(fsm, st))
end

