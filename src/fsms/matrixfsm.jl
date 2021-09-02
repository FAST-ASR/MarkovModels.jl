# SPDX-License-Identifier: MIT

"""
    struct MatrixFSM{T<:Semiring}
        π        # vector of initial probabilities
        T        # matrix of transition probabilities
        Tᵀ       # transpose of `A`
        C        # matrix mapping state -> pdfindex
        Cᵀ       # tranpose of `C`
    end

FSM where the arcs are stored as a sparse matrix. In addition to the
structure of the FSM, the `MatrixFSM` store the mapping between states
and `pdf-id` in the matrix `C`.

!!! warn
    In this format, the final state is just a "regular" states.
    Therefore, the dimension of `π` (similarly for `T` and `C`)
    will have one more dimension.

# Constructor

    MatrixFSM(fsm, pdfid_mapping)

"""
struct MatrixFSM{Tv} <: AbstractFSM{Tv}
    π::AbstractSparseVector{Tv}
    T::AbstractSparseMatrix{Tv}
    Tᵀ::AbstractSparseMatrix{Tv}
    C::AbstractSparseMatrix{Tv}
    Cᵀ::AbstractSparseMatrix{Tv}
    labels::Vector
end

function MatrixFSM(fsm::AbstractFSM{Tv}, pdfid_mapping::Dict,
                   keyfn::Function = identity) where Tv
    state2idx = Dict(s => i for (i, s) in enumerate(states(fsm)))
    S = length(fsm)

    π = spzeros(Tv, S+1)
    for s in filter(isinit, states(fsm))
        π[state2idx[s]] = s.initweight
    end

    T = spzeros(Tv, S+1, S+1)
    Tᵀ = spzeros(Tv, S+1, S+1)
    for src in states(fsm)
        for arc in arcs(fsm, src)
            T[state2idx[src], state2idx[arc.dest]] = arc.weight
            Tᵀ[state2idx[arc.dest], state2idx[src]] = arc.weight
        end
    end

    # Special case for the final states.
    for src in filter(isfinal, states(fsm))
        T[state2idx[src], end] = src.finalweight
        Tᵀ[end, state2idx[src]] = src.finalweight
    end
    T[end,end] = one(Tv)
    Tᵀ[end,end] = one(Tv)

    K = maximum(values(pdfid_mapping))
    C = spzeros(Tv, S+1, K+1)
    Cᵀ = spzeros(Tv, K+1, S+1)
    labels = Vector{Any}(undef, S+1)
    for s in states(fsm)
        C[state2idx[s], pdfid_mapping[keyfn(s.label)]] = one(Tv)
        Cᵀ[pdfid_mapping[keyfn(s.label)], state2idx[s]] = one(Tv)
        labels[state2idx[s]] = s.label
    end
    C[end,end] = one(Tv)
    Cᵀ[end,end] = one(Tv)
    labels[end] = nothing

    MatrixFSM(π, T, Tᵀ, C, Cᵀ, labels)
end

function states(fsm::MatrixFSM)
    S = length(fsm.π) - 1
    [State(i, fsm.labels[i], fsm.π[i], fsm.T[i,end]) for i in 1:S]
end

function arcs(fsm::MatrixFSM{Tv}, state) where Tv
    retval = []
    for (i, v) in zip(findnz(fsm.T[state.id, 1:end-1])...)
        dest = State(i, fsm.labels[i], fsm.π[i], fsm.T[i,end])
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
        copyto!(similar(mfsm.T, NT), mfsm.T),
        copyto!(similar(mfsm.Tᵀ, NT), mfsm.Tᵀ),
        copyto!(similar(mfsm.C, NT), mfsm.C),
        copyto!(similar(mfsm.Cᵀ, NT), mfsm.Cᵀ),
        mfsm.labels
    )
end
Base.convert(::Type{MatrixFSM{T}}, fsm::MatrixFSM{T}) where T<:Semiring = fsm

struct UnionMatrixFSM{Tv} <: AbstractFSM{Tv}
    ranges::Vector
    π::AbstractSparseVector{Tv}
    T::AbstractSparseMatrix{Tv}
    Tᵀ::AbstractSparseMatrix{Tv}
    C::AbstractSparseMatrix{Tv}
    Cᵀ::AbstractSparseMatrix{Tv}
    labels::Vector
end

function states(fsm::UnionMatrixFSM)
    states = []
    for (ri, r) in enumerate(fsm.ranges)
        for i in r[1]:r[end]-1
            s = State((ri, i), fsm.labels[i], fsm.π[i], fsm.T[i,r[end]])
            push!(states, s)
        end
    end
    states
end

function arcs(fsm::UnionMatrixFSM{Tv}, state) where Tv
    retval = []
    ri, i = state.id
    r = fsm.ranges[ri]
    for (j, v) in zip(findnz(fsm.T[i, r[1]:r[end]-1])...)
        j = r[1]+j-1
        dest = State((ri, j), fsm.labels[j], fsm.π[j], fsm.T[j,r[end]])
        push!(retval, Arc{Tv}(dest, fsm.T[i,j]))
    end
    retval
end


function Base.union(fsms::Vararg{MatrixFSM{Tv},N}) where {Tv,N}
    nstates_sofar = 0
    ranges = []
    for fsm in fsms
        S = length(fsm.π)
        push!(ranges, (1:S) .+ nstates_sofar)
        nstates_sofar += S
    end

    UnionMatrixFSM{Tv}(
        ranges,
        vcat(map(x -> x.π, fsms)...),
        blockdiag(map(x -> x.T, fsms)...),
        blockdiag(map(x -> x.Tᵀ, fsms)...),
        blockdiag(map(x -> x.C, fsms)...),
        blockdiag(map(x -> x.Cᵀ, fsms)...),
        vcat(map(x -> x.labels, fsms)...)
    )
end

