# SPDX-License-Identifier: MIT

#======================================================================
This file wrapped a few basic linear algebra operations in order to
have more fine-grained control between CPU/GPU code and different
semifield algebra operations.
======================================================================#

# Element-wise division.
eldiv!(out, x::AbstractArray, y::AbstractArray) = broadcast!(/, out, x, y)

# Element-wise multiplication.
elmul!(out, x::AbstractArray, y::AbstractArray) = broadcast!(*, out, x, y)
elmul!(out, x::CuSparseVector, y::AbstractVector) = elmul_svdv!(out, x, y)
elmul!(out, x::CuSparseVector, Y::AbstractMatrix) = elmul_svdm!(out, x, Y)

# Matrix multiplication.
matmul!(out, X::AbstractMatrix{T}, y::AbstractArray{T}) where T =
    mul!(out, X, y, one(T), zero(T))
matmul!(out, X::CuMatrix{T}, Y::CuMatrix{T}) where T =
    mul_dmdm!(out, X, Y)
matmul!(out, X::CuSparseMatrixCSR{T}, y::AbstractVector{T}) where T =
    mul_smdv!(out, X, y)
matmul!(out, X::CuSparseMatrixCSR{T}, Y::AbstractMatrix{T}) where T =
    mul_smdm!(out, X, Y)

