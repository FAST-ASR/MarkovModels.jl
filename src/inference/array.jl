# SPDX-License-Identifier: MIT

# Construct a block diagonal matrix with CUDA CSR matrix.
# Adapted from:
#   https://github.com/JuliaLang/julia/blob/1b93d53fc4bb59350ada898038ed4de2994cce33/stdlib/SparseArrays/src/sparsematrix.jl#L3396
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

function Base.vcat(X::CuSparseVector{T}...) where T
    num = length(X)
    nX = Int[ length(x) for x in X ]
    n = sum(nX)

    nnzX = Int[ nnz(x) for x in X ]
    nnz_total = sum(nnzX)
    iPtr = CuVector{Cint}(undef, nnz_total)
    nzVal = CuVector{T}(undef, nnz_total)

    nX_sofar = 0
    nnz_sofar = 0
    for i = 1:num
        iPtr[ (1:nnzX[i]) .+ nnz_sofar ] = X[i].iPtr .+ nX_sofar
        nzVal[ (1:nnzX[i]) .+ nnz_sofar ] = nonzeros(X[i])
        nX_sofar += nX[i]
        nnz_sofar += nnzX[i]
    end

    CuSparseVector{T}(iPtr, nzVal, n)
end
