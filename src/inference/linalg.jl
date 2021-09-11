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
function elmul!(out, x::AbstractSparseArray{T1}, y::AbstractArray{T2}) where {T1,T2}
    T = promote_type(T1, T2)
    fill!(out, zero(T))
    broadcast!(*, out, x, y)
end
elmul!(out, x::CuSparseVector, y::AbstractVector) = elmul_svdv!(out, x, y)
elmul!(out, x::CuSparseVector, Y::AbstractMatrix) = elmul_svdm!(out, x, Y)

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
