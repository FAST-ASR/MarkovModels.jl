# SPDX-License-Identifier: MIT

#======================================================================
Sparse matrix (CSC) and dense vector multiplication.
======================================================================#

function LinearAlgebra.mul!(c::CuVector{K}, A::CuSparseMatrixCSC{K}, b::CuVector{K},
                            α::Number, β::Number) where K
    @boundscheck size(A, 2) == size(b, 1) || throw(DimensionMismatch())
    @boundscheck size(A, 1) == size(c, 1) || throw(DimensionMismatch())

    if β != 1
        β != 0 ? rmul!(c, β) : fill!(c, zero(eltype(c)))
    end
    if length(A.nzVal) > 0
        ckernel = @cuda launch=false _cukernel_mul_smdv!(
            c,
            A.colPtr,
            A.rowVal,
            A.nzVal,
            b)
        config = launch_configuration(ckernel.fun)
        threads = min(length(b), config.threads)
        blocks = cld(length(b), threads)
        ckernel(c, A.colPtr, A.rowVal, A.nzVal, b)
    end
    c
end

function _cukernel_mul_smdv!(c, colPtr, rowVal, nzVal, b)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x

    for i = index:stride:length(b)
        for j = colPtr[i]:(colPtr[i+1]-1)
            c[rowVal[j]] += nzVal[j] * b[i]
        end
    end
    return
end

#======================================================================
Sparse matrix (CSC) transpose and dense vector multiplication.
======================================================================#

function LinearAlgebra.mul!(c::CuVector{K}, Aᵀ::Adjoint{K, <:CuSparseMatrixCSC},
                            b::CuVector{K}, α::Number, β::Number) where K
    @boundscheck size(Aᵀ, 2) == size(b, 1) || throw(DimensionMismatch())
    @boundscheck size(Aᵀ, 1) == size(c, 1) || throw(DimensionMismatch())

    A = parent(Aᵀ)
    if β != 1
        β != 0 ? rmul!(c, β) : fill!(c, zero(eltype(c)))
    end
    if length(A.nzVal) > 0
        ckernel = @cuda launch=false _cukernel_mul_smTdv!(
            c,
            A.colPtr,
            A.rowVal,
            A.nzVal,
            b)
        config = launch_configuration(ckernel.fun)
        threads = min(length(b), config.threads)
        blocks = cld(length(b), threads)
        ckernel(c, A.colPtr, A.rowVal, A.nzVal, b)
    end
    c
end

function _cukernel_mul_smTdv!(c, rowPtr, colVal, nzVal, b)
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
Sparse matrix (CSC) and dense matrix multiplication.
======================================================================#

function LinearAlgebra.mul!(C::CuMatrix{K}, A::CuSparseMatrixCSC{K}, B::CuMatrix{K},
              α::Number, β::Number) where K
    @boundscheck size(A, 2) == size(B, 1) || throw(DimensionMismatch())
    @boundscheck size(A, 1) == size(C, 1) || throw(DimensionMismatch())
    @boundscheck size(B, 2) == size(C, 2) || throw(DimensionMismatch())

    if β != 1
        β != 0 ? rmul!(C, β) : fill!(C, zero(eltype(C)))
    end
    if length(A.nzVal) > 0
        ckernel = @cuda launch=false _cukernel_mul_smdm!(
            C,
            A.colPtr,
            A.rowVal,
            A.nzVal,
            B)
        config = launch_configuration(ckernel.fun)
        threads = min(size(C, 2), config.threads)
        blocks = cld(size(C, 2), threads)
        ckernel(C, A.colPtr, A.rowVal, A.nzVal, B)
    end
    C
end

function _cukernel_mul_smdm!(C, colPtr, rowVal, nzVal, B)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x

    for i = index:stride:size(C, 2)
        for j = 1:size(B, 1)
            for k = colPtr[j]:(colPtr[j+1]-1)
                C[rowVal[k], i] += nzVal[k] * B[j, i]
            end
        end
    end
    return
end

#======================================================================
Sparse matrix (CSC) transposed and dense matrix multiplication.
======================================================================#


function LinearAlgebra.mul!(C::CuMatrix{K}, Aᵀ::Adjoint{K, <:CuSparseMatrixCSC},
                            B::CuMatrix{K}, α::Number, β::Number) where K
    @boundscheck size(Aᵀ, 2) == size(B, 1) || throw(DimensionMismatch())
    @boundscheck size(Aᵀ, 1) == size(C, 1) || throw(DimensionMismatch())
    @boundscheck size(B, 2) == size(C, 2) || throw(DimensionMismatch())

    A = parent(Aᵀ)
    if β != 1
        β != 0 ? rmul!(C, β) : fill!(C, zero(eltype(C)))
    end
    if length(A.nzVal) > 0
        ckernel = @cuda launch=false _cukernel_mul_smTdm!(
            C,
            A.colPtr,
            A.rowVal,
            A.nzVal,
            B)
        config = launch_configuration(ckernel.fun)
        threads = min(size(C, 1), config.threads)
        blocks = cld(size(C, 1), threads)
        ckernel(C, A.colPtr, A.rowVal, A.nzVal, B)
    end
    C
end

function _cukernel_mul_smTdm!(C, rowPtr, colVal, nzVal, B)
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
Sparse vector and dense veector broadcasting.
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
        ckernel(dest, f, nzVal, nzInd, y)
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
