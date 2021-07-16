# SPDX-License-Identifier: MIT

const N_THREADS_1D = 256
const N_THREADS_2D = (16, 16)

#######################################################################
# elmul_svdm!
#
#   Broadcasted element-wise multiplication of a sparse vector and a
#   dense matrix.
#

function _cukernel_elmul_svdm!(out, nzVal, nzInd, dsMat)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    for i = index:stride:length(nzInd)
        idx = nzInd[i]
        for j in 1:size(dsMat, 2)
            out[idx, j] = nzVal[i] * dsMat[idx,j ]
        end
    end
    return
end

function elmul_svdm!(out::AbstractMatrix{T}, x::CuSparseVector{T},
                     M::AbstractMatrix{T}) where T
    size(out) == size(M) || throw(DimensionMistmatch())
    length(x) == size(M, 1) || throw(DimensionMistatch())
    fill!(out, zero(T))
    I = SparseArrays.nonzeroinds(x)
    V = nonzeros(x)
    N = length(x.nzVal)

    if N > 0
        numblocks = ceil(Int, N/N_THREADS_1D)
        @cuda threads=N_THREADS_1D blocks=numblocks _cukernel_elmul_svdm!(out, V, I, M)
    end
    return out
end

#######################################################################
# elmul_svdv!
#
#   Element-wise multiplication of a sparse vector and a dense vector.
#

function _cukernel_elmul_svdv!(out, nzVal, nzInd, dsVec)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    for i = index:stride:length(nzInd)
        idx = nzInd[i]
        out[idx] = nzVal[i] * dsVec[idx]
    end
    return
end

function elmul_svdv!(out::CuVector, x::CuSparseVector{T}, y::CuVector{T}) where T
    size(out) == size(y) || throw(DimensionMistmatch())
    length(x) == length(y) || throw(DimensionMistatch())
    fill!(out, zero(T))
    I = SparseArrays.nonzeroinds(x)
    V = nonzeros(x)
    N = length(x.nzVal)
    if N > 0
        numblocks = ceil(Int, N/N_THREADS_1D)
        @cuda threads=N_THREADS_1D blocks=numblocks _cukernel_elmul_svdv!(out, V, I, y)
    end
    return out
end

#######################################################################
# mul_smdv!
#
#   Product of a sparse matrix and a dense vector.
#

function _cukernel_mul_smdv!(out, N, rowPtr, colVal, nzVal, x)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    for ti = index:stride:N
        for i = rowPtr[ti]:(rowPtr[ti+1]-1)
            out[ti] += nzVal[i] * x[colVal[i]]
        end
    end
    return
end

function mul_smdv!(out::CuVector{T}, M::CuSparseMatrixCSR{T}, x::CuVector{T}) where T
    fill!(out, zero(T))
    rowPtr = M.rowPtr
    colVal = M.colVal
    nzVal = M.nzVal
    N = size(M,1)
    if N > 0
        numblocks = ceil(Int, N/N_THREADS_1D)
        @cuda threads=N_THREADS_1D blocks=numblocks _cukernel_mul_smdv!(out, N, rowPtr,
                                                                        colVal, nzVal, x)
    end
    return out
end

#######################################################################
# mul_smdm!
#
#   Product of a sparse matrix and a dense matrix.
#

function _cukernel_mul_smdm!(out, N, rowPtr, colVal, nzVal, Y)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x

    for ti = index:stride:N
        for i = rowPtr[ti]:(rowPtr[ti+1]-1)
            for j in 1:size(Y, 2)
                out[ti,j] += nzVal[i] * Y[colVal[i],j]
            end
        end
    end
    return
end

function mul_smdm!(out::AbstractMatrix{T}, X::CuSparseMatrixCSR{T},
                   Y::AbstractMatrix{T}) where T
    fill!(out, zero(T))
    rowPtr = X.rowPtr
    colVal = X.colVal
    nzVal = X.nzVal
    N = size(X, 1)
    if N > 0
        numblocks = ceil(Int, N/N_THREADS_1D)
        @cuda threads=N_THREADS_1D blocks=numblocks _cukernel_mul_smdm!(out, N, rowPtr,
                                                                        colVal, nzVal, Y)
    end
    return out
end

#######################################################################
# mul_dmdm!
#
#   Product of a dense matrix and a dense matrix.
#

function _cukernel_mul_dmdm!(out, X, Y)
    index_i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride_i = blockDim().x * gridDim().x
    index_j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    stride_j = blockDim().y * gridDim().y

    for i = index_i:stride_i:size(X, 1)
        for j in index_j:stride_j:size(Y, 2)
            for k in 1:size(X, 2)
                out[i,j] += X[i,k] * Y[k,j]
            end
        end
    end
    return
end

function mul_dmdm!(out::AbstractMatrix{T}, X::CuMatrix{T},
                   Y::CuMatrix{T}) where T
    fill!(out, zero(T))
    M, N = size(X, 1), size(Y, 2)
    numblocks = (
        ceil(Int, M/N_THREADS_2D[1]),
        ceil(Int, N/N_THREADS_2D[2])
    )
    @cuda threads=N_THREADS_2D blocks=numblocks _cukernel_mul_dmdm!(out, X, Y)
    return out
end

#######################################################################
