# MarkovModels.jl
#
# Lucas Ondel, 2021

struct InitStateID end
const initstateid = InitStateID()
Base.show(io::IO, ::InitStateID) = print(io, "initstateid")

struct FinalStateID end
const finalstateid = FinalStateID()
Base.show(io::IO, ::FinalStateID) = print(io, "finalstateid")

const StateID = Union{UInt64, InitStateID, FinalStateID}
const PdfIndex = Union{UInt64,Nothing}
const Label = Union{String,Nothing}

struct State
    id::StateID
    pdfindex::PdfIndex
end

"""
    isemitting(state)

Returns `true` if `state` a pdf index associated.
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


struct Link{T<:SemiField}
    dest::State
    ilabel::Label
    olabel::Label
    weight::T
end

"""
    islabeled(link)

Returns `true` if the `link` has a label.
"""
hasinputlabel(l::Link) = ! isnothing(l.ilabel)
hasoutputlabel(l::Link) = ! isnothing(l.olabel)

const shared_initstate = State(initstateid, nothing)
const shared_finalstate = State(finalstateid, nothing)

mutable struct StateIDCounter
    count::UInt64
end

"""
    struct FSM{T}
        ...
    end

Structure of a FSM. The type `T` indicates the type of the arcs'
weight.
"""
struct FSM{T<:SemiField}
    idcounter::StateIDCounter
    states::Set{State}
    links::Dict{State, Vector{Link{T}}}
end
FSM{T}() where T = FSM{T}(StateIDCounter(0), Set{State}(), Dict{State, Vector{Link{T}}}())

"""
    addstate!(fsm[; id, pdfindex])

Add `state` to `fsm` and return it.
"""
function addstate!(fsm::FSM{T}; id = nothing, pdfindex = nothing) where T
    if isnothing(id)
        fsm.idcounter.count += 1
        id = fsm.idcounter.count
    else
        fsm.idcounter.count = id
    end
    s = State(id, pdfindex)
    push!(fsm.states, s)
    s
end

"""
    link!(fsm::FSM{T}, src, dest[, weight = zero(T)])

Add a weighted connection between `state1` and `state2`.
"""
function link!(fsm::FSM{T}, src::State, dest::State; ilabel::Label = nothing,
               olabel::Label = nothing, weight::T = one(T)) where T
    list = get(fsm.links, src, Link{T}[])
    push!(list, Link{T}(dest, ilabel, olabel, weight))
    fsm.links[src] = list
end

"""
    LinearFSM([T = LogSemiField{Float64}, ]seq[, emissionsmap = Dict()])

Create a linear FSM of type `T` from a sequence of labels `seq`. If
`emissionsmap` is provided, every item `l` of `seq` with a matching entry
in `emissionsmap` will be assigned the pdf index `emissionsmap[l]`.
"""
function LinearFSM(T::Type{<:SemiField}, sequence, emissionsmap = Dict())
    fsm = FSM{T}()
    prevstate = initstate(fsm)
    for token in sequence
        pdfindex = get(emissionsmap, token, nothing)
        s = addstate!(fsm, pdfindex = pdfindex)
        link!(fsm, prevstate, s, token)
        prevstate = s
    end
    link!(fsm, prevstate, finalstate(fsm))
    fsm
end
LinearFSM(sequence, emissionsmap = Dict()) =
    LinearFSM(LogSemiField{Float64}, sequence, emissionsmap)

"""
    initstate(fsm)

Returns the initial state of `fsm`.
"""
initstate(fsm::FSM) = shared_initstate

"""
    finalstate(fsm)

Returns the final state of `fsm`.
"""
finalstate(fsm::FSM) = shared_finalstate

"""
    states(fsm)

Iterator over the state of `fsm`.
"""
states(fsm::FSM) = [initstate(fsm), finalstate(fsm), fsm.states...]

"""
    links(fsm, state)

Iterator over the links to the children (i.e. next states) of `state`.
"""
links(fsm::FSM{T}, state::State) where T<:SemiField = get(fsm.links, state, Link{T}[])

