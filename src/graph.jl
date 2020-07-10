# Implementation of a full expanded graph, i.e. all its states and arcs
# are stored explictly.

export addstate!
export link!
export State
export Graph
export LinearGraph

struct State <: AbstractState
    id::StateID
    pdfindex::Union{Int64, Nothing}
    outgoing::Vector{Link}
    incoming::Vector{Link}
    name::String
end
State(id, pdfindex = nothing) = State(id, pdfindex, Vector{Link}(), Vector{Link}(),
                                      string(id))
State(id, pdfindex, name) = State(id, pdfindex, Vector{Link}(), Vector{Link}(),
                                  name)
id(s::State) = s.id
pdfindex(s::State) = s.pdfindex
children(s::State) = s.outgoing
parents(s::State) = s.incoming
name(s::State) = s.name


struct GraphState <: AbstractState
    id::StateID
    subgraph::AbstractGraph
    entrystate::State
    exitstate::State
    outgoing::Vector{Link}
    incoming::Vector{Link}
end

function GraphState(id, subgraph::AbstractGraph)
    subgraph = deepcopy(subgraph)
    entry, exit = initstate(subgraph), finalstate(subgraph)
    incoming = Vector{Link}()
    outgoing = Vector{Link}()
    GraphState(id, subgraph, entry, exit, incoming, outgoing)
end

id(s::GraphState) = s.id
pdfindex(s::GraphState) = nothing
children(s::GraphState) = s.outgoing
parents(s::GraphState) = s.incoming
subchildren(s::GraphState) = initstate(s.subgraph).outgoing
subparents(s::GraphState) = finalstate(s.subgraph).incoming

# Override the default behavior of `emittingstates`.
emittingstates(::Forward, s::GraphState) = EmittingStatesIterator(s, subchildren)
emittingstates(::Backward, s::GraphState) = EmittingStatesIterator(s, subparents)


struct Graph <: AbstractGraph
    states::Dict{StateID, State}
    Graph() = new(Dict{StateID, State}(initstateid => State(initstateid),
                                       finalstateid => State(finalstateid)))
end

#######################################################################
# Methods to construct the graph.

"""
    addstate!(graph, state)

Add `state` to `graph`.
"""
addstate!(g::Graph, s::State) = g.states[s.id] = s

"""
    link!(state1, state2, weight)

Add a weighted connection between `state1` and `state2`.
"""
function link!(s1::State, s2::State, weight::Real = 0.)
    push!(s1.outgoing, Link(s2, weight))
    push!(s2.incoming, Link(s1, weight))
end

#######################################################################
# Implementation of the AbstractGraph interface.

initstate(g::Graph) = g.states[initstateid]
finalstate(g::Graph) = g.states[finalstateid]
states(g::Graph) = values(g.states)

struct ArcIterator
    siter
end

function Base.iterate(iter::ArcIterator, iterstate = nothing)
    # Initialize the state of the iterator.
    if iterstate == nothing
        state, siterstate = iterate(iter.siter)
        citer = children(state)
        next = iterate(citer)
    else
        state, siterstate, citer, citerstate = iterstate
        next = iterate(citer, citerstate)
    end

    while next == nothing
        nextstate = iterate(iter.siter, siterstate)

        # Finished iterating over the states.
        # End the iterations.
        if nextstate == nothing return nothing end

        state, siterstate = nextstate
        citer = children(state)
        next = iterate(citer)
    end

    link, citerstate = next
    newiterstate = (state, siterstate, citer, citerstate)
    newarc = (state, link.dest, link.weight)

    return newarc, newiterstate
end

arcs(g::Graph) = ArcIterator(states(g))

#######################################################################
# Create a linear graph from a sequence of symbol

function LinearGraph(sequence::AbstractArray{String},
                     emissionsmap::Dict{String, <:Integer})
    g = Graph()
    prevstate = initstate(g)
    for (i, token) in enumerate(sequence)
        s = addstate!(g, State(i, emissionsmap[token], token))
        link!(prevstate, s, 0.)
        prevstate = s
    end
    link!(prevstate, finalstate(g), 0.)
    g
end

