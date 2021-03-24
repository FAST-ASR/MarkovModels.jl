# MarkovModels.jl
#
# Lucas Ondel, 2020

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
    states::Dict{StateID, State{T}}

    FSM{T}() where T = new{T}(
        StateIDCounter(0),
        Dict{StateID, State{T}}(
            initstateid => State{T}(initstateid),
            finalstateid => State{T}(finalstateid)
        ),
    )
end

"""
    addstate!(fsm[, pdfindex = ..., label = "..."])

Add `state` to `fsm` and return it.
"""
function addstate!(fsm::FSM{T}; id = nothing, pdfindex = nothing, label = nothing) where T
    if isnothing(id)
        fsm.idcounter.count += 1
        id = fsm.idcounter.count
    else
        fsm.idcounter.count = max(fsm.idcounter.count, id)
    end
    s = State{T}(id, pdfindex, label, Vector{Link{State{T},T}}())
    fsm.states[s.id] = s
end

"""
    link!(src, dest[, weight = 0])

Add a weighted connection between `state1` and `state2`. By default,
`weight = 0`.
"""
link!(src::State{T}, dest::State{T}, weight::T = one(T)) where T =
    push!(src.links, Link(dest, weight))


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
        s = addstate!(fsm, pdfindex = pdfindex, label = token)
        link!(prevstate, s)
        prevstate = s
    end
    link!(prevstate, finalstate(fsm))
    fsm
end
LinearFSM(sequence, emissionsmap = Dict()) =
    LinearFSM(LogSemiField{Float64}, sequence, emissionsmap)

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

"""
    states(fsm)

Iterator over the state of `fsm`.
"""
states(fsm::FSM) = values(fsm.states)

