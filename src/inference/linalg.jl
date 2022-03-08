# SPDX-License-Identifier: MIT

#======================================================================
This file wrapped a few basic linear algebra operations in order to
have more fine-grained control between CPU/GPU code and different
semifield algebra operations.
======================================================================#

## Element-wise division.

eldiv!(out, x::AbstractArray, y::AbstractArray) = broadcast!(/, out, x, y)

## Element-wise multiplication.

elmul!(out, x::AbstractArray, y::AbstractArray) =  broadcast!(*, out, x, y)
elmul!(out, x::CuSparseVector, y::AbstractVector) = elmul_svdv!(out, x, y)
elmul!(out, x::CuSparseVector, Y::AbstractMatrix) = elmul_svdm!(out, x, Y)
function elmul!(out, x::AbstractSparseArray{T1}, y::AbstractArray{T2}) where {T1,T2}
    T = promote_type(T1, T2)
    fill!(out, zero(T))
    broadcast!(*, out, x, y)
end
elmul!(out, x::AbstractSparseMatrix{T1}, y::AbstractArray{T2}) where {T1,T2} = _elmul_sm_dm!(out, x, y)
elmul!(out, x::AbstractSparseMatrix{T1}, y::AbstractSparseMatrix{T2}) where {T1,T2} = _elmul_sm_dm!(out, x, y)
elmul!(out, x::AbstractArray{T1}, y::AbstractSparseMatrix{T2}) where {T1,T2} = _elmul_sm_dm!(out, y, x)

function _elmul_sm_dm!(out, X::AbstractSparseMatrix{T1}, y::AbstractArray{T2}) where {T1,T2}
    T = promote_type(T1, T2)
    fill!(out, zero(T))
	if sparsity_level(X) > 0.1
        return broadcast!(*, out, X, y)
    end
    m, n = size(X)
    length(y) in [m, n, m*n] || throw(DimensionMismatch("arrays could not be broadcast to a common size; got a matrix ($m, $n) and $(length(y))"))
    rows = rowvals(X)
    vals = nonzeros(X)
    for j = 1:n
       for i in nzrange(X, j)
          row = rows[i]
          val = vals[i]
          out[row, j] = val * _getindex(y, row, j)
       end
    end
    return out
end

sparsity_level(x::AbstractSparseArray{T}) where T = begin
    nnz(x) / reduce(*, size(x))
end
_getindex(x::AbstractMatrix{T}, i, j) where T = begin
    if size(x, 1) == 1
        x[1, j]
    elseif size(x, 2) == 1
        x[i, 1]
    else
        x[i, j]
    end
end
_getindex(x::AbstractVector{T}, i, j) where T = x[i]


## Matrix multiplication.

function matmul!(out, X::AbstractMatrix{T1}, y::AbstractArray{T2}) where {T1,T2}
    T = promote_type(T1, T2)
    mul!(out, X, y, one(T), zero(T))
end

function matmul!(out, X::AbstractSparseMatrix{T1}, y::AbstractArray{T2}) where {T1,T2}
    T = promote_type(T1, T2)
    fill!(out, zero(T))
    mul!(out, X, y, one(T), zero(T))
end

matmul!(out, X::CuMatrix, Y::CuMatrix) = mul_dmdm!(out, X, Y)
matmul!(out, X::CuSparseMatrixCSR, y::AbstractVector) = mul_smdv!(out, X, y)
matmul!(out, X::CuSparseMatrixCSR, Y::AbstractMatrix) = mul_smdm!(out, X, Y)
