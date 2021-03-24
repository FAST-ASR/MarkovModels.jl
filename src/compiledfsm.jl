# MarkovModels.jl
#
# Lucas Ondel, 2021
#

# Compute the distance of each emitting state to the final state.
function calculate_distance(fsm::FSM{T}) where T<:SemiField
    fsmᵀ = fsm |> transpose
    distances = Dict()
    visited = Set()
    stack = [(initstate(fsmᵀ), 0)]
    while ! isempty(stack)
        state, steps = popfirst!(stack)
        for (nstate, _) in nextemittingstates(state)
            cdist = get(distances, nstate, typemax(typeof(steps)))
            distances[nstate] = min(cdist, steps + 1)
            if nstate ∉ visited
                push!(visited, nstate)
                push!(stack, (nstate, steps + 1))
            end
        end
    end
    distances
end

"""
    struct CompiledFSM{T<:AlgebraicStructure}
        distances::AbstractVector{Int64}
        state_pdf::AbstractMatrix{Int}
        A::AbstractMatrix{T}
        π::AbstractMatrix{T}
        ω::AbstractMatrix{T}
    end

A compiled FSM is a compact representation of an FSM suitable for
efficient inference aglorithms.
"""
struct CompiledFSM{T<:SemiField}
    distances::AbstractVector{Int64}
    state_pdf::AbstractMatrix{T}
    A::AbstractMatrix{T}
    π::AbstractVector{T}
    ω::AbstractVector{T}
end

nstates(cfsm::CompiledFSM) = size(cfsm.state_pdf, 1)

"""
    compile([T = LogSemifield{Float64}], fsm)

Compile an FSM to have a representation more adapted for inference
algorithms. Note that once an FSM is compiled, it can not longer
be represented as a graph in the Jupyter notebook.
"""
function compile(fsm::FSM{T}) where T<:SemiField
    statemap = Dict(s => i for (i,s) in enumerate(filter(isemitting, collect(states(fsm)))))
    startstates = Dict(s => w for (s,w) in nextemittingstates(initstate(fsm)))
    endstates = Dict(s => w for (s,w) in nextemittingstates(initstate(transpose(fsm))))

    S = length(statemap)

    distances = zeros(Int64, S)
    for (s, d) in calculate_distance(fsm)
        distances[statemap[s]] = d
    end

    pdfmap = Dict()
    for s in keys(statemap)
        statelist = get(pdfmap, s.pdfindex, [])
        push!(statelist, s)
        pdfmap[s.pdfindex] = statelist
    end

    P = length(pdfmap)

    state_pdf = spzeros(T, S, P)
    for i in 1:P
        i ∉ keys(pdfmap) && continue
        for s in pdfmap[i]
            state_pdf[statemap[s],i] = one(T)
        end
    end

    A = sparse(zeros(T, S, S))
    for s in keys(statemap)
        for (ns, w) in nextemittingstates(s)
            A[statemap[s], statemap[ns]] = w
        end
    end

    π = sparse(zeros(T, S))
    for (s, w) in startstates
        π[statemap[s]] = w
    end

    ω = sparse(zeros(T, S))
    for (s, w) in endstates
        ω[statemap[s]] = w
    end

    CompiledFSM(distances, state_pdf, A, π, ω)
end

