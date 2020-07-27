
module HiddenMarkovModel


#######################################################################
# State interface.

export AbstractState
export id
export initstateid
export isemitting
export finalstateid
export Link
export name
export pdfindex
export StateID

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
    StateID

Type of the state identifier.
"""
const StateID = Union{Int64, InitStateID, FinalStateID, Missing}

"""
    AbstractState

Abstract type of a graph state (vertex).
"""
abstract type AbstractState end

Base.:(==)(s1::AbstractState, s2::AbstractState) = id(s1) == id(s2)

Base.show(io::IO, s::AbstractState) = print(io, "State(id = $(id(s)), pdfindex = $(pdfindex(s)), $(name(s)))")

"""
    id(state)

Returns the identifier of the state.
"""
id

"""
    name(state)

Returns the name of the state.
"""
pdfindex

"""
    pdfindex(state)

Returns the index of the pdf the state is connected with. Returns
`nothing` is the state is non-emitting.
"""
pdfindex

"""
    isemitting(state)

Returns `true` is `state` is associated with a pdf.
"""
isemitting(s::AbstractState) = pdfindex(s) ≢ nothing

"""
    struct Link{T} where T <: AbstractFloat

Weighted link pointing to a state `dest`.
"""
struct Link{T<:AbstractFloat}
    dest::AbstractState
    weight::T
end

#######################################################################
# Graph interface.

export AbstractGraph
export arcs
export children
export initstate
export finalstate
export parents
export states

"""
    AbstractGraph

Type representing a graph.
"""
abstract type AbstractGraph end

"""
    initstate(graph)

Return the initial state of the graph.
"""
initstate

"""
    finalstate(graph)

Return the final state of the graph.
"""
finalstate

"""
    states(graph)

Return an iterator over the states of the graph.
"""
states

"""
    arcs(graph)

Return an iterator over the arcs of the graph.
"""
arcs

"""
    children(state)

Returns an iterator over the children. For each element, the iterator
returns the child and the weight associated with.
"""
children

"""
    parents(state)

Returns an iterator over the parents. For each element, the iterator
returns the parent and the weight associated with.
"""
parents


#######################################################################
# Emitting state iterator

export Forward
export forward
export Backward
export backward
export emittingstates

struct Forward end
const forward = Forward()

struct Backward end
const backward = Backward()

struct EmittingStatesIterator
    state::AbstractState
    getlinks::Function
end

emittingstates(::Forward, s::AbstractState) = EmittingStatesIterator(s, children)
emittingstates(::Backward, s::AbstractState) = EmittingStatesIterator(s, parents)

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

#######################################################################
# Pretty display of the graph in IJulia.

function Base.show(io, ::MIME"image/svg+xml", g::AbstractGraph)
    dotpath, dotfile = mktemp()
    svgpath, svgfile = mktemp()

    write(dotfile, "Digraph {\n")
    write(dotfile, "rankdir=LR;")

    for state in states(g)
        shape = isemitting(state) ? "circle" : "point"
        label = "$(id(state)):$(name(state))"
        write(dotfile, "$(id(state)) [ shape=\"$(shape)\", label=\"$(label)\" ];\n")
    end
    for arc in arcs(g)
        src, dest, weight = id(arc[1]), id(arc[2]), round(arc[3], digits = 3)
        write(dotfile, "$(src) -> $(dest) [ label=\"$(weight)\" ];\n")
    end
    write(dotfile, "}\n")
    close(dotfile)
    run(`dot -Tsvg $(dotpath) -o $(svgpath)`)

    xml = read(svgfile, String)
    write(io, xml)

    close(svgfile)

    rm(dotpath)
    rm(svgpath)
end

#######################################################################
# Concrete graph implementation

export PruningStrategy
export ThresholdPruning
export nopruning

include("graph.jl")

#######################################################################
# Pretty display the sparse matrix (i.e. from αβrecursion).

import Printf:@sprintf
function Base.show(io::IO, ::MIME"text/plain", a::Array{Dict{State, T},1}) where T <: AbstractFloat
    for n in 1:length(a)
        write(io, "[n = $n]  \t")
        max = foldl(((sa,wa), (s,w)) -> wa < w ? (s,w) : (sa,wa), a[n]; init=first(a[n]))
        write(io, first(max) |> name)
        for (s, w) in sort(a[n]; by=x->name(x))
            write(io, "\t$(id(s)):$(name(s)) = $(@sprintf("%.3f", w))  ")
        end
        write(io, "\n")
    end
end

#######################################################################
# Major algorithms

include("algorithms.jl")


#######################################################################
# Other

include("../src/misc.jl")

end
