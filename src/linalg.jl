# SPDX-License-Identifier: MIT


#Base.similar(x::CuSparseVector, T::Type) =
#    CuSparseVector{T}(copy(CUSPARSE.nonzeroinds(x)), similar(nonzeros(x)), x.dims[1])
#Base.similar(M::CuSparseMatrixCSR, T::Type) =
#    CuSparseMatrixCSR{T}(copy(M.rowPtr), copy(M.colVal), similar(nonzeros(M), T), M.dims)

#======================================================================
This file wrapped a few basic linear algebra operations in order to
have more fine-grained control between CPU/GPU code and different
semifield algebra operations.
======================================================================#

# Element-wise division.
eldiv!(out, x::AbstractArray, y::AbstractArray) = broadcast!(/, out, x, y)

# Element-wise multiplication.
elmul!(out, x::AbstractArray, y::AbstractArray) =  broadcast!(*, out, x, y)
function elmul!(out, x::AbstractSparseArray{T}, y::AbstractArray{T}) where T
    fill!(out, zero(T))
    broadcast!(*, out, x, y)
end
elmul!(out, x::CuSparseVector, y::AbstractVector) = elmul_svdv!(out, x, y)
elmul!(out, x::CuSparseVector, Y::AbstractMatrix) = elmul_svdm!(out, x, Y)

# Matrix multiplication.
matmul!(out, X::AbstractMatrix{T}, y::AbstractArray{T}) where T =
    mul!(out, X, y, one(T), zero(T))
function matmul!(out, X::AbstractSparseMatrix{T}, y::AbstractArray{T}) where T
    fill!(out, zero(T))
    mul!(out, X, y, one(T), zero(T))
end
matmul!(out, X::CuMatrix{T}, Y::CuMatrix{T}) where T =
    mul_dmdm!(out, X, Y)
matmul!(out, X::CuSparseMatrixCSR{T}, y::AbstractVector{T}) where T =
    mul_smdv!(out, X, y)
matmul!(out, X::CuSparseMatrixCSR{T}, Y::AbstractMatrix{T}) where T =
    mul_smdm!(out, X, Y)

# Construct a block diagonal matrix with CUDA CSR matrix.
# Adapted from:
#   https://github.com/JuliaLang/julia/blob/1b93d53fc4bb59350ada898038ed4de2994cce33/stdlib/SparseArrays/src/sparsematrix.jl#L3396
function blockdiag(X::CuSparseMatrixCSR{T}...) where T
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
