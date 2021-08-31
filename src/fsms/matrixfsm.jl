# SPDX-License-Identifier: MIT

"""
    struct MatrixFSM{T<:Semiring}
        π        # vector of initial probabilities
        ω        # vector of final probabilities
        T        # matrix of transition probabilities
        Tᵀ       # transpose of `A`
        C        # matrix mapping state -> pdfindex
        Cᵀ       # tranpose of `C`
    end

FSM where the arcs are stored as a sparse matrix. In addition to the
structure of the FSM, the `MatrixFSM` store the mapping between states
and `pdf-id` in the matrix `C`.

# Constructor

    MatrixFSM(fsm, pdfid_mapping)

"""
struct MatrixFSM{Tv} <: AbstractFSM{Tv}
    π::AbstractSparseVector{Tv}
    ω::AbstractSparseVector{Tv}
    T::AbstractSparseMatrix{Tv}
    Tᵀ::AbstractSparseMatrix{Tv}
    C::AbstractSparseMatrix{Tv}
    Cᵀ::AbstractSparseMatrix{Tv}
    labels::Vector
end

function MatrixFSM(fsm::AbstractFSM{Tv}, pdfid_mapping::Dict,
                   keyfn::Function = identity) where Tv
    state2idx = Dict(s => i for (i, s) in enumerate(states(fsm)))
    S = length(state2idx)

    π = spzeros(Tv, S)
    for s in filter(isinit, states(fsm))
        π[state2idx[s]] = s.initweight
    end

    ω = spzeros(Tv, S)
    for s in filter(isfinal, states(fsm))
        ω[state2idx[s]] = s.finalweight
    end

    T = spzeros(Tv, S, S)
    Tᵀ = spzeros(Tv, S, S)
    for src in states(fsm)
        for arc in arcs(fsm, src)
            T[state2idx[src], state2idx[arc.dest]] = arc.weight
            Tᵀ[state2idx[arc.dest], state2idx[src]] = arc.weight
        end
    end

    K = maximum(values(pdfid_mapping))
    C = spzeros(Tv, S, K)
    Cᵀ = spzeros(Tv, K, S)
    labels = Vector{Any}(undef, S)
    for s in states(fsm)
        C[state2idx[s], pdfid_mapping[keyfn(s.label)]] = one(Tv)
        Cᵀ[pdfid_mapping[keyfn(s.label)], state2idx[s]] = one(Tv)
        labels[state2idx[s]] = s.label
    end

    MatrixFSM(π, ω, T, Tᵀ, C, Cᵀ, labels)
end

function states(fsm::MatrixFSM)
    [State(i, fsm.labels[i], fsm.π[i], fsm.ω[i])
     for i in 1:length(fsm.π)]
end

function arcs(fsm::MatrixFSM{Tv}, state) where Tv
    retval = []
    for (i, v) in zip(findnz(fsm.T[state.id, :])...)
        dest = State(i, fsm.labels[i], fsm.π[i], fsm.ω[i])
        push!(retval, Arc{Tv}(dest, v))
    end
    retval
end

"""
    gpu(cfsm)

Move the compiled fsm `cfsm` to GPU.
"""
function gpu(cfsm::MatrixFSM{Tv}) where Tv
    T = CuSparseMatrixCSC(cfsm.T)
    Tᵀ = CuSparseMatrixCSC(cfsm.Tᵀ)
    C = CuSparseMatrixCSC(cfsm.C)
    Cᵀ = CuSparseMatrixCSC(cfsm.Cᵀ)
    return MatrixFSM{Tv}(
        CuSparseVector(cfsm.π),
        CuSparseVector(cfsm.ω),
        CuSparseMatrixCSR(Tᵀ.colPtr, Tᵀ.rowVal, Tᵀ.nzVal, T.dims),
        CuSparseMatrixCSR(T.colPtr, T.rowVal, T.nzVal, Tᵀ.dims),
        CuSparseMatrixCSR(Cᵀ.colPtr, Cᵀ.rowVal, Cᵀ.nzVal, C.dims),
        CuSparseMatrixCSR(C.colPtr, C.rowVal, C.nzVal, Cᵀ.dims),
        cfsm.labels
    )
end

function Base.convert(::Type{MatrixFSM{NT}}, mfsm) where NT <: Semiring
    MatrixFSM{NT}(
        copyto!(similar(mfsm.π, NT), mfsm.π),
        copyto!(similar(mfsm.ω, NT), mfsm.ω),
        copyto!(similar(mfsm.T, NT), mfsm.T),
        copyto!(similar(mfsm.Tᵀ, NT), mfsm.Tᵀ),
        copyto!(similar(mfsm.C, NT), mfsm.C),
        copyto!(similar(mfsm.Cᵀ, NT), mfsm.Cᵀ),
        mfsm.labels
    )
end
Base.convert(::Type{MatrixFSM{T}}, fsm::MatrixFSM{T}) where T<:Semiring = fsm

function Base.union(fsms::Vararg{MatrixFSM{Tv},N}) where {Tv,N}
    labels = []
    for fsm in fsms
        for label in fsm.labels
            push!(labels, label)
        end
    end

    MatrixFSM{Tv}(
        vcat(map(x -> x.π, fsms)...),
        vcat(map(x -> x.ω, fsms)...),
        blockdiag(map(x -> x.T, fsms)...),
        blockdiag(map(x -> x.Tᵀ, fsms)...),
        blockdiag(map(x -> x.C, fsms)...),
        blockdiag(map(x -> x.Cᵀ, fsms)...),
        labels
    )
end

