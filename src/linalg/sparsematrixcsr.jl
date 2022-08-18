# SPDX-License-Identifier: MIT

#=

Compressed sparse row data structure. The  implementation is an
adaptation of the Julia sparse matrix (CSC):
https://github.com/JuliaLang/julia/blob/742b9abb4dd4621b667ec5bb3434b8b3602f96fd/stdlib/SparseArrays/src/sparsematrix.jl

=#

"""
    SparseMatrixCSR{Tv,Ti<:Integer} <: AbstractSparseMatrix{Tv,Ti}

Matrix type for storing sparse matrices in the
[Compressed Sparse Row](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format))
format.
"""
struct SparseMatrixCSR{Tv, Ti, Trp<:AbstractVector{Ti},Tcv<:AbstractVector{Ti},
                       Tnv<:AbstractVector{Tv}} <: AbstractSparseMatrix{Tv,Ti}
    m::Int       # Number of rows.
    n::Int       # Number of columns.
    rowptr::Trp  # Row i is in rowptr[i]:(rowptr[i+1]-1).
    colval::Tcv  # Column indices of stored values.
    nzval::Tnv   # Non-zeros values.

    function SparseMatrixCSR{Tv,Ti,Trp,Tcv,Tnv}(m::Integer, n::Integer, rowptr::Trp,
                                    colval::Tcv, nzval::Tnv) where {Tv,
                                                                    Ti<:Int,
                                                                    Trp<:AbstractVector{Ti},
                                                                    Tcv<:AbstractVector{Ti},
                                                                    Tnv<:AbstractVector{Tv}}
        m, n = promote(m, n)
        new(m, n, rowptr, colval, nzval)
    end
end

#======================================================================
Constructors.
======================================================================#

#SparseVectorSI(n::Integer, nzind::AbstractVector{Ti}, nzval::AbstractVector{Tv}) where {Ti,Tv} =
#    SparseVectorSI{Tv,Ti}(n, nzind, nzval)

#======================================================================
AbstractArray interface.
======================================================================#

#size(x::SparseVectorSI) = (x.n,)
#
#function getindex(x::SparseVectorSI, j)
#    @boundscheck checkbounds(x, i)
#end
#
#count(f, x::SparseVector) = count(f, nonzeros(x)) + f(zero(eltype(x)))*(length(x) - nnz(x))

