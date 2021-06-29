# SPDX-License-Identifier: MIT

function _cukernel_spare_dense_mul!(out, sp_nzVal, ds_val)
    i = threadIdx().x
    out[i] = sp_nzVal[i] * ds_val[i]
    return
end

# Element wise vector multiplication.
function el_mul(x::CuSparseVector{T}, y::CuVector{T}) where T
    out = similar(x)
    ds_val = y[x.iPtr]
    sp_nzVal = x.nzVal
    out_nzVal = out.nzVal
    @cuda threads=length(sp_nzVal) _cukernel_spare_dense_mul!(out_nzVal, sp_nzVal,
                                                              ds_val)
    return out
end

function el_mul!(out::CuVector, x::CuSparseVector{T}, y::CuVector{T}) where T
    I = SparseArrays.nonzeroinds(x)
    ds_val = y[I]
    sp_nzVal = x.nzVal
    out_nzVal = view(out, I)
    @cuda threads=length(sp_nzVal) _cukernel_spare_dense_mul!(out_nzVal, sp_nzVal,
                                                              ds_val)
    return out
end

function _cukernel_smdv!(out, cols, rows, vals, x)
    ti = threadIdx().x
    i = rows[ti]
    j = cols[ti]
    out[i] += vals[ti] * x[j]
    return
end

function smdv(M::CuSparseMatrixCOO{T}, x::CuVector{T}) where T
    out = similar(x, size(M, 1))
    fill!(out, zero(T))
    cols = M.colInd
    rows = M.rowInd
    vals = M.nzVal
    @cuda threads=length(vals) _cukernel_smdv!(out, cols, rows, vals, x)
    return out
end

function _cukernel_csr_smdv!(out, N, rowPtr, colVal, nzVal, x)
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

function smdv!(out::CuVector{T}, M::CuSparseMatrixCSR{T}, x::CuVector{T}) where T
    fill!(out, zero(T))
    rowPtr = M.rowPtr
    colVal = M.colVal
    nzVal = M.nzVal
    N = size(M,1)
    numblocks = ceil(Int, N/256)
    #@show N, numblocks
    @cuda threads=256 blocks=numblocks _cukernel_csr_smdv!(out, N, rowPtr, colVal, nzVal, x)
    return out
end
