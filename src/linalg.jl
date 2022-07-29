# SPDX-License-Identifier: MIT

const SparseAdjOrTrans{K} = Union{Adjoint{K, <:AbstractSparseMatrix{K}},
                                  Transpose{K, <:AbstractSparseMatrix{K}}} where K
const AnySparseMatrix{K} = Union{SparseAdjOrTrans{K},
                                 AbstractSparseMatrix{K}} where K

const CuSparseAdjOrTrans{K} = Union{Adjoint{K, <:CuSparseMatrix},
                                    Transpose{K, <:CuSparseMatrix}} where K

const IndexRange = Union{UnitRange, Colon}

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

function LinearAlgebra.mul!(C::SparseLowRankMatrix{K}, A::AnySparseMatrix{K},
                            B::AnySparseMatrix{K}, α::Number, β::Number) where K
    @boundscheck size(A, 2) == size(B, 1) || throw(DimensionMismatch())
    @boundscheck size(A, 1) == size(C, 1) || throw(DimensionMismatch())
    @boundscheck size(B, 2) == size(C, 2) || throw(DimensionMismatch())

    if β != 1
        β != 0 ? rmul!(C, β) : fill!(C, zero(eltype(C)))
    end
    if length(A.nzVal) > 0
        ckernel = @cuda launch=false _cukernel_mul_smdm!(
            C,
            A.rowPtr,
            A.colVal,
            A.nzVal,
            B)
        config = launch_configuration(ckernel.fun)
        threads = min(size(C, 1), config.threads)
        blocks = cld(size(C, 1), threads)
        ckernel(C, A.rowPtr, A.colVal, A.nzVal, B; threads, blocks)
    end
    C
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

#=====================================================================
LIL (LIst of List) matrix format.
======================================================================#

struct LILMatrix{K, Tv <: AbstractVector{K}} <: AbstractMatrix{K}
    numrows::Int64
    cols::Vector{Tv}
end

Base.size(M::LILMatrix) = (M.numrows, length(M.cols))

Base.getindex(M::LILMatrix, i::Int, j::Int) = M.cols[j][i]
Base.getindex(M::LILMatrix, i::IndexRange, j::Int) = M.cols[j]
Base.setindex!(M::LILMatrix, v, i::Int, j::Int) = M.cols[j][i] = v
Base.setindex!(M::LILMatrix, v, i::IndexRange, j::Int) = M.cols[j] = v

function lilzeros(K::Type{<:Any}, numrows::Int, numcols::Int)
    LILMatrix(numrows, repeat([spzeros(K, numrows)], numcols))
end
lilzeros(numrows, numcols) = lilzeros(Float64, numrows, numcols)

#======================================================================
Conversion from/to CSR/CSC matrix with semiring elements.
In all cases, it is necessary to convert the underlying data to
classical floating point value to call the CUDA API.
======================================================================#

function CUDA.CUSPARSE.CuSparseMatrixCSR(M::CuSparseMatrixCSC{K}) where K <: Semiring
    T = K.parameters[1]

    tmp_csc = CuSparseMatrixCSC{T}(
        M.colPtr,
        M.rowVal,
        convert(CuVector{T}, M.nzVal),
        M.dims
    )
    tmp_csr = CUDA.CUSPARSE.CuSparseMatrixCSR(tmp_csc)

    CUDA.CUSPARSE.CuSparseMatrixCSR{K}(
        tmp_csr.rowPtr,
        tmp_csr.colVal,
        convert(CuVector{K}, tmp_csr.nzVal),
        tmp_csr.dims
    )
end


function CUDA.CUSPARSE.CuSparseMatrixCSC(M::CuSparseMatrixCSR{K}) where K <: Semiring
    T = K.parameters[1]

    tmp_csr = CuSparseMatrixCSR{T}(
        M.rowPtr,
        M.colVal,
        convert(CuVector{T}, M.nzVal),
        M.dims
    )
    tmp_csc = CUDA.CUSPARSE.CuSparseMatrixCSC(tmp_csr)

    CUDA.CUSPARSE.CuSparseMatrixCSC{K}(
        tmp_csc.colPtr,
        tmp_csc.rowVal,
        convert(CuVector{K}, tmp_csc.nzVal),
        tmp_csc.dims
    )
end

#======================================================================
Materialize an adjoint/transpose of a CSR/CSC matrix.
======================================================================#

function Base.copy(Mᵀ::Union{Adjoint{K, <:CuSparseMatrixCSR},
                             Transpose{K, <:CuSparseMatrixCSR}}) where K <:Semiring
    M = parent(Mᵀ)
    CuSparseMatrixCSR(CuSparseMatrixCSC(M.rowPtr, M.colVal, M.nzVal,
                                        reverse(M.dims)))
end

function Base.copy(Mᵀ::Union{Adjoint{K, <:CuSparseMatrixCSC},
                             Transpose{K, <:CuSparseMatrixCSC}}) where K <:Semiring
    M = parent(Mᵀ)
    CuSparseMatrixCSC(CuSparseMatrixCSR(M.colPtr, M.rowVal, M.nzVal,
                                        reverse(M.dims)))
end

#======================================================================
Building a block diagonal matrix from CUDA sparse matrices.
======================================================================#

function SparseArrays.blockdiag(X::CuSparseMatrixCSC{K}...) where K
    num = length(X)
    mX = Int[ size(x, 1) for x in X ]
    nX = Int[ size(x, 2) for x in X ]
    m = sum(mX)
    n = sum(nX)

    colPtr = CuVector{Cint}(undef, n+1)
    nnzX = Int[ nnz(x) for x in X ]
    nnz_res = sum(nnzX)
    rowVal = CuVector{Cint}(undef, nnz_res)
    nzVal = CuVector{K}(undef, nnz_res)

    nnz_sofar = 0
    nX_sofar = 0
    mX_sofar = 0
    for i = 1:num
        colPtr[ (1:nX[i]+1) .+ nX_sofar ] = X[i].colPtr .+ nnz_sofar
        rowVal[ (1:nnzX[i]) .+ nnz_sofar ] = X[i].rowVal .+ mX_sofar
        nzVal[ (1:nnzX[i]) .+ nnz_sofar ] = nonzeros(X[i])
        nnz_sofar += nnzX[i]
        nX_sofar += nX[i]
        mX_sofar += mX[i]

    end
    CUDA.@allowscalar colPtr[n+1] = nnz_sofar + 1

    CuSparseMatrixCSC{K}(colPtr, rowVal, nzVal, (m, n))
end

function SparseArrays.blockdiag(X::CuSparseMatrixCSR{T}...) where T
    num = length(X)
    mX = Int[ size(x, 1) for x in X ]
    nX = Int[ size(x, 2) for x in X ]
    m = sum(mX) # number of rows
    n = sum(nX) # number of cols

    rowPtr = CuVector{Cint}(undef, m+1)
    nnzX = Int[ nnz(x) for x in X ]
    nnz_res = sum(nnzX)
    colVal = CuVector{Cint}(undef, nnz_res)
    nzVal = CuVector{T}(undef, nnz_res)

    nnz_sofar = 0
    nX_sofar = 0
    mX_sofar = 0
    for i = 1:num
        rowPtr[ (1:mX[i]+1) .+ mX_sofar ] = X[i].rowPtr .+ nnz_sofar
        colVal[ (1:nnzX[i]) .+ nnz_sofar ] = X[i].colVal .+ nX_sofar
        nzVal[ (1:nnzX[i]) .+ nnz_sofar ] = nonzeros(X[i])
        nnz_sofar += nnzX[i]
        nX_sofar += nX[i]
        mX_sofar += mX[i]

    end
    CUDA.@allowscalar rowPtr[m+1] = nnz_sofar + 1

    CuSparseMatrixCSR{T}(rowPtr, colVal, nzVal, (m, n))
end

#======================================================================
Vertical concatenation of CUDA sparse vectors.
======================================================================#

function Base.vcat(X::CuSparseVector{K}...) where K
    num = length(X)
    nX = Int[length(x) for x in X]
    n = sum(nX)

    nnzX = Int[nnz(x) for x in X]
    nnz_total = sum(nnzX)
    iPtr = CuVector{Cint}(undef, nnz_total)
    nzVal = CuVector{K}(undef, nnz_total)

    nX_sofar = 0
    nnz_sofar = 0
    for i = 1:num
        iPtr[ (1:nnzX[i]) .+ nnz_sofar ] = X[i].iPtr .+ nX_sofar
        nzVal[ (1:nnzX[i]) .+ nnz_sofar ] = nonzeros(X[i])
        nX_sofar += nX[i]
        nnz_sofar += nnzX[i]
    end

    CuSparseVector{K}(iPtr, nzVal, n)
end

#======================================================================
Sparse matrix (CSR) and dense vector multiplication.
======================================================================#

function LinearAlgebra.mul!(c::CuVector{K}, A::CuSparseMatrixCSR{K},
                            b::CuVector{K}, α::Number, β::Number) where K
    @boundscheck size(A, 2) == size(b, 1) || throw(DimensionMismatch())
    @boundscheck size(A, 1) == size(c, 1) || throw(DimensionMismatch())

    if β != 1
        β != 0 ? rmul!(c, β) : fill!(c, zero(eltype(c)))
    end
    if length(A.nzVal) > 0
        ckernel = @cuda launch=false _cukernel_mul_smdv!(
            c,
            A.rowPtr,
            A.colVal,
            A.nzVal,
            b)
        config = launch_configuration(ckernel.fun)
        threads = min(length(c), config.threads)
        blocks = cld(length(c), threads)
        ckernel(c, A.rowPtr, A.colVal, A.nzVal, b; threads, blocks)
    end
    c
end

LinearAlgebra.mul!(c::CuVector{K}, Aᵀ::CuSparseAdjOrTrans{K},
                   b::CuVector{K}, α::Number, β::Number) where K =
    mul!(c, copy(Aᵀ), b, α, β)


function _cukernel_mul_smdv!(c, rowPtr, colVal, nzVal, b)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x

    for i = index:stride:length(c)
        for j = rowPtr[i]:(rowPtr[i+1]-1)
            c[i] += nzVal[j] * b[colVal[j]]
        end
    end
    return
end

#======================================================================
Sparse matrix (CSR) and dense matrix multiplication.
======================================================================#


function LinearAlgebra.mul!(C::CuMatrix{K}, A::CuSparseMatrixCSR{K},
                            B::CuMatrix{K}, α::Number, β::Number) where K
    @boundscheck size(A, 2) == size(B, 1) || throw(DimensionMismatch())
    @boundscheck size(A, 1) == size(C, 1) || throw(DimensionMismatch())
    @boundscheck size(B, 2) == size(C, 2) || throw(DimensionMismatch())

    if β != 1
        β != 0 ? rmul!(C, β) : fill!(C, zero(eltype(C)))
    end
    if length(A.nzVal) > 0
        ckernel = @cuda launch=false _cukernel_mul_smdm!(
            C,
            A.rowPtr,
            A.colVal,
            A.nzVal,
            B)
        config = launch_configuration(ckernel.fun)
        threads = min(size(C, 1), config.threads)
        blocks = cld(size(C, 1), threads)
        ckernel(C, A.rowPtr, A.colVal, A.nzVal, B; threads, blocks)
    end
    C
end

LinearAlgebra.mul!(C::CuMatrix{K}, Aᵀ::CuSparseAdjOrTrans{K},
                   B::CuMatrix{K}, α::Number, β::Number) where K =
    mul!(C, copy(Aᵀ), B, α, β)

function _cukernel_mul_smdm!(C, rowPtr, colVal, nzVal, B)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x

    for i = index:stride:size(C, 1)
        for j = 1:size(C, 2)
            for k = rowPtr[i]:(rowPtr[i+1]-1)
                C[i, j] += nzVal[k] * B[colVal[k], j]
            end
        end
    end
    return
end


#======================================================================
Sparse vector and dense vector broadcasting.
======================================================================#

const SupportedOperator = Union{typeof(*), typeof(/)}

function _copyto!(f::SupportedOperator, dest::CuArray{K}, x::CuSparseVector{K},
                  y::CuVector{K}) where K

    fill!(dest, zero(K))
    nzInd = SparseArrays.nonzeroinds(x)
    nzVal = nonzeros(x)
    nnz = length(nzVal)

    if nnz > 0
        ckernel = @cuda launch=false _cukernel_bc_svdv!(
            dest,
            f,
            nzVal,
            nzInd,
            y)
        config = launch_configuration(ckernel.fun)
        threads = min(nnz, config.threads)
        blocks = cld(nnz, threads)
        ckernel(dest, f, nzVal, nzInd, y; threads, blocks)
    end
    dest
end

_copy(f::SupportedOperator, x::CuSparseVector{K}, y::CuVector{K}) where K =
    _copyto!(f, similar(y), x, y)

function _cukernel_bc_svdv!(dest, f, nzVal, nzInd, y)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x

    for i = index:stride:length(nzInd)
        dest[nzInd[i]] = f(nzVal[i], y[nzInd[i]])
    end
    return
end

function Broadcast.copy(bc::Broadcasted{CUDA.CUSPARSE.CuSparseVecStyle})
    bcf = Broadcast.flatten(bc)
    _copy(bcf.f, bcf.args...)
end

function Broadcast.copyto!(dest::CuArray, bc::Broadcasted{CUDA.CUSPARSE.CuSparseVecStyle})
    bcf = Broadcast.flatten(bc)
    _copyto!(bcf.f, dest, bcf.args...)
end

