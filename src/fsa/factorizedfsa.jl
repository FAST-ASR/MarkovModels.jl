# SPDX-License-Identifier: MIT

#======================================================================
Sparse Low-Rank matrix, i.e. a matrix given by the sum of a sparse and
a low-rank matrix.
======================================================================#

struct SparseLowRankMatrix{K,
                           TS <: AbstractSparseMatrix{K},
                           TD <: AbstractMatrix{K},
                           TU <: AbstractMatrix{K},
                           TV<: AbstractMatrix{K}
	                      } <: AbstractMatrix{K}
    S::TS
    D::TD
	U::TU
	V::TV
end

Base.copy(M::SparseLowRankMatrix) = M.S + M.U * (I + M.D) * M.V'

SparseArrays.sparse(M::Union{Adjoint{<:Number, <:SparseLowRankMatrix},
                             Transpose{<:Number, <:SparseLowRankMatrix}}) =
    SparseLowRankMatrix(sparse(parent(M).S'), sparse(parent(M).D'),
                        parent(M).V, parent(M).U)

Base.size(M::SparseLowRankMatrix) = size(M.S)

Base.getindex(M::SparseLowRankMatrix{K}, i::Int, j::Int) where K =
    M.S[i, j] + dot((M.U * (I + M.D))[i, :], M.V[j, :])
Base.getindex(M::SparseLowRankMatrix, i::IndexRange, j::IndexRange) =
    SparseLowRankMatrix(M.S[i, j], M.D[i, j], M.U[i, :], M.V[j, :])
Base.getindex(M::SparseLowRankMatrix, i::IndexRange, j) =
    M.S[i, j] + (M.U * (I + D) * M.V')[i, j]

function SparseArrays.blockdiag(Ms::SparseLowRankMatrix...)
    SparseLowRankMatrix(
        blockdiag(Any[M.S for M in Ms]...),
        blockdiag(Any[M.D for M in Ms]...),
        blockdiag(Any[M.U for M in Ms]...),
        blockdiag(Any[M.V for M in Ms]...),
    )
end

Base.:*(A::SparseLowRankMatrix, B::AnySparseMatrix) =
    SparseLowRankMatrix(A.S * B, A.D, A.U, B' * A.V)
Base.:*(A::AnySparseMatrix, B::SparseLowRankMatrix) =
	SparseLowRankMatrix(A * B.S, B.D, A * B.U, B.V)

Base.:+(A::SparseLowRankMatrix, B::AnySparseMatrix) =
    SparseLowRankMatrix(A.S + B, A.D, A.U, A.V)
Base.:+(A::AnySparseMatrix, B::SparseLowRankMatrix) = B + A

function Base.hcat(A::SparseLowRankMatrix{K,TS,TD, TU,TV},
                   v::AbstractVector{K}) where {
                        K,
                        TS<:AbstractSparseMatrix{K},
                        TD<:AbstractMatrix{K},
                        TU<:AbstractMatrix{K},
                        TV<:AbstractMatrix{K}
                    }
    pad = fill!(similar(v, 1, size(A.V, 2)), zero(eltype(A)))
    SparseLowRankMatrix(
        hcat(A.S, v),
        A.D,
        A.U,
        vcat(A.V, pad)
    )
end

function Base.vcat(A::SparseLowRankMatrix{K,TS,TU,TV},
                   v::AbstractMatrix{K}) where {
                        K,
                        TS<:AbstractSparseMatrix{K},
                        TU<:AbstractMatrix{K},
                        TV<:AbstractMatrix{K}
                    }
    pad = fill!(similar(v, 1, size(A.U, 2)), zero(eltype(A)))
    SparseLowRankMatrix(
        vcat(A.S, v),
        A.D,
        vcat(A.U, pad),
        A.V,
    )
end

#======================================================================
FSA with a sparse low-rank transition matrix.
======================================================================#

const FactorizedFSA{K} = FSA{K, <:SparseLowRankMatrix} where K

function FactorizedFSA(initws, arcs, finalws, λ)
    # Get the semiring of the FSM.
    K = typeof(initws[1][2])

    # Get the set of states indices.
    states = reduce(
        union,
        [
            Set(map(first, initws)),
            Set(filter(x -> x > 0, map(x -> x.first[1], arcs))),
            Set(filter(x -> x > 0, map(x -> x.first[2], arcs))),
            Set(map(first, finalws))
        ]
    )
    nstates = length(states)

    # Count the epsilon states.
    eps_states = reduce(
        union,
        [
            Set(filter(x -> x <= 0, map(x -> x.first[1], arcs))),
            Set(filter(x -> x <= 0, map(x -> x.first[2], arcs))),
        ]
    )
    n_eps = length(eps_states)

    I_S, J_S, V_S = [], [], K[]
    I_D, J_D, V_D = [], [], K[]
    I_U, J_U, V_U = [], [], K[]
    I_V, J_V, V_V = [], [], K[]
    epsilon_nodes = Set()
    for a in arcs
        if minimum(a.first) > 0
            push!(I_S, a.first[1])
            push!(J_S, a.first[2])
            push!(V_S, a.second)
        elseif a.first[1] <= 0 && a.first[2] <= 0
            # arc from and to an epsilon node
            push!(I_D, 1 - a.first[1])
            push!(J_D, 1 - a.first[2])
            push!(V_D, a.second)
        elseif a.first[1] <= 0
            # outgoing arc from epsilon node
            push!(J_V, 1 - a.first[1])
            push!(I_V, a.first[2])
            push!(V_V, a.second)
        else
            # incoming arc to epsilon node
            push!(I_U, a.first[1])
            push!(J_U, 1 - a.first[2])
            push!(V_U, a.second)
        end
    end
    S = sparse(I_S, J_S, V_S, nstates, nstates)
    U = sparse(I_U, J_U, V_U, nstates, n_eps)
    V = sparse(I_V, J_V, V_V, nstates, n_eps)
    D = sparse(I_D, J_D, V_D, n_eps, n_eps)
    T = SparseLowRankMatrix(S, D, U, V)

    FSA(
        sparsevec(map(x -> x[1], initws), map(x -> x[2], initws), nstates),
        T,
        sparsevec(map(x -> x[1], finalws), map(x -> x[2], finalws), nstates),
        λ
    )
end

function write_arcs!(file, fsa::FactorizedFSA)
    n_eps = size(fsa.T.U, 2)
    for i in (1:n_eps) .+ (nstates(fsa) + 1)
        write(file, "$i [ label=\"ϵ\" shape=circle style=filled ];\n")
    end

    write_arcs!(file, fsa.T.U; dest_offset = nstates(fsa) + 1)
    write_arcs!(file, copy(fsa.T.V'); src_offset = nstates(fsa) + 1)
    write_arcs!(file, copy(fsa.T.D); src_offset = nstates(fsa) + 1,
                dest_offset = nstates(fsa) + 1)
    write_arcs!(file, fsa.T.S)
end

