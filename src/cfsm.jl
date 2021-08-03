# SPDX-License-Identifier: MIT

"""
    struct CompiledFSM{T<:Semifield}
        π        # vector of initial probabilities
        ω        # vector of final probabilities
        T        # matrix of transition probabilities
        Tᵀ       # transpose of `A`
        C        # matrix mapping state -> pdfindex
        Cᵀ       # tranpose of `C`
    end

Compiled FSM: matrix/vector format of an FSM used by inference
algorithms. All the fields are stored in sparse containers.
"""
struct CompiledFSM{SF<:Semifield}
    π::AbstractSparseVector{SF}
    ω::AbstractSparseVector{SF}
    T::AbstractSparseMatrix{SF}
    Tᵀ::AbstractSparseMatrix{SF}
    C::AbstractSparseMatrix{SF}
    Cᵀ::AbstractSparseMatrix{SF}
end

function Base.show(io::IO, cfsm::CompiledFSM)
    nstates = length(cfsm.π)
    narcs = length(nonzeros(cfsm.T))
    print(io, "$(typeof(cfsm)) # states: $nstates # arcs: $narcs")
end

"""
    compile(fsm, K)

Compile `fsm` into a inference-friendly format [`CompiledFSM`](@ref).
`K` is the total number of emission pdfs. Note that the fsm, is not
requested to use all the pdf indices.

!!! warning
    This function assumes that all states of `fsm` are associated to a
    pdf index.

"""
function compile(fsm::FSM{SF}, K::Integer) where SF
    allstates = collect(states(fsm))
    S = length(allstates)

    # Initial states' probabilities.
    π = spzeros(SF, S)
    for s in filter(isinit, allstates) π[s.id] = s.initweight end

    # Final states' probabilities.
    ω = spzeros(SF, S)
    for s in filter(isfinal, allstates) ω[s.id] = s.finalweight end

    # Transition matrix.
    T = spzeros(SF, S, S)
    Tᵀ = spzeros(SF, S, S)
    for src in allstates
        for arc in arcs(fsm, src)
            T[src.id, arc.dest.id] = arc.weight
            Tᵀ[arc.dest.id, src.id] = arc.weight
        end
    end

    # Connection matrix.
    C = spzeros(SF, S, K)
    Cᵀ = spzeros(SF, K, S)
    for s in allstates
        C[s.id, s.pdfindex] = one(SF)
        Cᵀ[s.pdfindex, s.id] = one(SF)
    end

    CompiledFSM{SF}(π, ω, T, Tᵀ, C, Cᵀ)
end

"""
    gpu(cfsm)

Move the compiled fsm `cfsm` to GPU.
"""
function gpu(cfsm::CompiledFSM{SF}) where SF
    T = CuSparseMatrixCSC(cfsm.T)
    Tᵀ = CuSparseMatrixCSC(cfsm.Tᵀ)
    C = CuSparseMatrixCSC(cfsm.C)
    Cᵀ = CuSparseMatrixCSC(cfsm.Cᵀ)
    return CompiledFSM{SF}(
        CuSparseVector(cfsm.π),
        CuSparseVector(cfsm.ω),
        CuSparseMatrixCSR(Tᵀ.colPtr, Tᵀ.rowVal, Tᵀ.nzVal, T.dims),
        CuSparseMatrixCSR(T.colPtr, T.rowVal, T.nzVal, Tᵀ.dims),
        CuSparseMatrixCSR(Cᵀ.colPtr, Cᵀ.rowVal, Cᵀ.nzVal, C.dims),
        CuSparseMatrixCSR(C.colPtr, C.rowVal, C.nzVal, Cᵀ.dims),
    )
end

function Base.convert(::Type{CompiledFSM{NT}}, cfsm) where NT <: Semifield
    CompiledFSM{NT}(
        copyto!(similar(cfsm.π, NT), cfsm.π),
        copyto!(similar(cfsm.ω, NT), cfsm.ω),
        copyto!(similar(cfsm.T, NT), cfsm.T),
        copyto!(similar(cfsm.Tᵀ, NT), cfsm.Tᵀ),
        copyto!(similar(cfsm.C, NT), cfsm.C),
        copyto!(similar(cfsm.Cᵀ, NT), cfsm.Cᵀ),
    )
end
Base.convert(::Type{T}, cfsm::T) where T <: CompiledFSM = cfsm

struct UnionCompiledFSM{SF<:Semifield,U}
    cfsm::CompiledFSM{SF}
end

function Base.show(io::IO, ucfsm::UnionCompiledFSM)
    nstates = length(ucfsm.cfsm.π)
    narcs = length(nonzeros(ucfsm.cfsm.T))
    print(io, "$(typeof(ucfsm)) # states: $nstates # arcs: $narcs")
end

function Base.union(cfsms::CompiledFSM{SF}...) where SF
    U = length(cfsms)
    UnionCompiledFSM{SF,U}(
        CompiledFSM{SF}(
            vcat(map(x -> x.π, cfsms)...),
            vcat(map(x -> x.ω, cfsms)...),
            blockdiag(map(x -> x.T, cfsms)...),
            blockdiag(map(x -> x.Tᵀ, cfsms)...),
            blockdiag(map(x -> x.C, cfsms)...),
            blockdiag(map(x -> x.Cᵀ, cfsms)...),
        )
    )
end

function gpu(ucfsm::UnionCompiledFSM{SF,U}) where {SF,U}
    UnionCompiledFSM{SF,U}(ucfsm.cfsm |> gpu)
end
