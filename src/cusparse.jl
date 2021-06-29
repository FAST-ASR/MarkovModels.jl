# SPDX-License-Identifier: MIT

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
    I = SparseArrays.nonzeroinds(x)
    V = nonzeros(x)
    N = length(x.nzVal)
    numblocks = ceil(Int, N/256)
    @cuda threads=256 blocks=numblocks _cukernel_elmul_svdv!(out, V, I, y)
    return out
end

#######################################################################


#######################################################################
# mul_smdv!
#
#   Product of a sparse matrix and a dense vector.
#

function _cukernel_mul_smdv!(out, N, rowPtr, colVal, nzVal, x)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    #@cushow threadIdx().x, blockIdx().x, blockDim().x, index
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
    numblocks = ceil(Int, N/256)
    #@show N, numblocks
    @cuda threads=256 blocks=numblocks _cukernel_mul_smdv!(out, N, rowPtr, colVal, nzVal, x)
    return out
end

#######################################################################
