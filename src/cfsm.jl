# SPDX-License-Identifier: MIT

"""
    struct CompiledFSM{T<:Semifield}
        π        # vector of initial probabilities
        ω        # vector of final probabilities
        A        # matrix of transition probabilities
        Aᵀ       # transpose of `A`
        pdfmap   # mapping state -> pdfindex
    end

Compiled FSM: matrix/vector format of an FSM used by inference
algorithms. Note that in this form, every state is an emitting state.
"""
struct CompiledFSM{T<:Semifield}
    π::AbstractVector{T}
    ω::AbstractVector{T}
    A::AbstractMatrix{T}
    Aᵀ::AbstractMatrix{T}
    pdfmap::AbstractVector{PdfIndex}
end

cfsmtype(::CompiledFSM{T}) where T<:Semifield = T

"""
    compile(fsm; allocator = spzeros)

Compile `fsm` into a inference-friendly format [`CompiledFSM`](@ref).
`allocator` is a function analogous to `zeros` which creates an
N-dimensional array and fills it with zero elements.
"""
function compile(fsm::FSM{T}; allocator = spzeros) where T
    allstates = collect(states(fsm))
    S = length(allstates)

    # Initial states' probabilities.
    π = allocator(T, S)
    for s in filter(isinit, allstates) π[s.id] = s.initweight end

    # Final states' probabilities.
    ω = allocator(T, S)
    for s in filter(isfinal, allstates) ω[s.id] = s.finalweight end

    # Transition matrix.
    A = allocator(T, S, S)
    Aᵀ = allocator(T, S, S)
    for src in allstates
        for link in links(fsm, src)
            A[src.id, link.dest.id] = link.weight
            Aᵀ[link.dest.id, src.id] = link.weight
        end
    end

    pdfmap = [s.pdfindex for s in sort(allstates, by = p -> p.id)]

    CompiledFSM{T}(π, ω, A, Aᵀ, pdfmap)
end

"""
    gpu(cfsm)

Move the compiled fsm `cfsm` to GPU.
"""
function gpu(cfsm::CompiledFSM{T}) where T
    if ! issparse(cfsm.π)
        return CompiledFSM{T}(
            CuArray(cfsm.π),
            CuArray(cfsm.ω),
            CuArray(cfsm.A),
            CuArray(cfsm.Aᵀ),
            CuArray(cfsm.pdfmap)
        )
    end

    A = CuSparseMatrixCSC(cfsm.A)
    Aᵀ = CuSparseMatrixCSC(cfsm.Aᵀ)
    return CompiledFSM{T}(
        CuSparseVector(cfsm.π),
        CuSparseVector(cfsm.ω),
        CuSparseMatrixCSR(Aᵀ.colPtr, Aᵀ.rowVal, Aᵀ.nzVal, A.dims),
        CuSparseMatrixCSR(A.colPtr, A.rowVal, A.nzVal, A.dims),
        CuArray(cfsm.pdfmap)
    )
end

function Base.convert(::Type{CompiledFSM{NT}}, cfsm) where NT <: Semifield
    CompiledFSM(
        copyto!(similar(cfsm.π, NT), cfsm.π),
        copyto!(similar(cfsm.ω, NT), cfsm.ω),
        copyto!(similar(cfsm.A, NT), cfsm.A),
        copyto!(similar(cfsm.Aᵀ, NT), cfsm.Aᵀ),
        copyto!(similar(cfsm.pdfmap, Int), cfsm.pdfmap)
    )
end
Base.convert(::Type{T}, cfsm::T) where T <: CompiledFSM = cfsm
