# SPDX-License-Identifier: MIT

"""
    PartialVector{Tv} <: AbstractVector{Tv}

Vector of size `n` containted in a buffer of size `q` where `q >= n`.
When multiply with a matrix, only the matching dimension of the matrix
will be considered.
"""
struct PartialVector{Tv, TV<:AbstractArray{Tv}} <: AbstractVector{Tv}
	val::TV
	n::Int64

    function PartialVector(val, n)
        new{eltype(val),typeof(val)}(val, n)
    end
end

Adapt.adapt_storage(to::Type{CuArray}, x::PartialVector) =
    PartialVector(adapt(to, x.val), x.n)
Adapt.adapt_storage(to::Type{Array}, x::PartialVector) =
    PartialVector(adapt(to, x.val), x.n)

#======================================================================
Array interface.
======================================================================#

Base.size(x::PartialVector) = size(x.val)
function Base.getindex(x::PartialVector{Tv}, i::Int) where Tv
    i > x.n && return zero(Tv)
    x.val[i]
end

#======================================================================
Matrix-vector multiplication. For a partial vector with `N` non-zeros
elements, we perform the following operation:

    M[1:n, 1:n] * x[1:n]

Note that this implicitely assumes that `M` is a squared matrix.
======================================================================#

function Base.:*(M::AbstractMatrix, x::PartialVector)
    T = promote_type(eltype(M), eltype(x.val))
    y = fill!(similar(x.val, size(M, 1)), zero(T))
    y[1:x.n] = M[1:x.n,1:x.n] * x.val[1:x.n]
    y
end

function LinearAlgebra.mul!(c::CuVector{K},
                            A::CuSparseMatrixCSR{K},
                            b::PartialVector{K}) where K
    @boundscheck size(A, 2) == size(b, 1) || throw(DimensionMismatch())
    @boundscheck size(A, 1) == size(c, 1) || throw(DimensionMismatch())

    fill!(c, zero(K))
    if length(A.nzVal) > 0
        ckernel = @cuda launch=false _cukernel_mul_smpv!(
            c,
            A.rowPtr,
            A.colVal,
            A.nzVal,
            b.val,
            b.n)
        config = launch_configuration(ckernel.fun)
        threads = min(b.n, config.threads)
        blocks = cld(b.n, threads)
        ckernel(c, A.rowPtr, A.colVal, A.nzVal, b.val, b.n; threads, blocks)
    end
    c
end

function _cukernel_mul_smpv!(c, rowPtr, colVal, nzVal, bval, bn)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x

    for i = index:stride:bn
        for j = rowPtr[i]:(rowPtr[i+1]-1)
            if colVal[j] <= bn
                c[i] += nzVal[j] * bval[colVal[j]]
            end
        end
    end
    return
end

function elmul!(f::SupportedOperator, dest::CuArray{K}, x::PartialVector{K},
                  y::CuArray{K}) where K

    fill!(dest, zero(K))

    if x.n > 0
        ckernel = @cuda launch=false _cukernel_bc_svpv!(
            dest,
            f,
            x.val,
            x.n,
            y)
        config = launch_configuration(ckernel.fun)
        threads = min(x.n, config.threads)
        blocks = cld(x.n, threads)
        ckernel(dest, f, x.val, x.n, y; threads, blocks)
    end
    dest
end

function _cukernel_bc_svpv!(dest, f, xval, xn,  y)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x

    for i = index:stride:xn
        dest[i] = f(xval[i], y[i])
    end
    return
end


"""
    PartialMatrix{Tv} <: AbstractMatrix{Tv}

Matrix partially filled.
"""
struct PartialMatrix{Tv, TV<:AbstractArray{Tv}} <: AbstractMatrix{Tv}
	val::TV
    batchLength::Vector{Int}
    batchCumSize::Vector{Int}

    function PartialMatrix(val, batchLength, batchCumSize)
        new{eltype(val),typeof(val)}(val, batchLength, batchCumSize)
    end
end

Adapt.adapt_storage(to::Type{CuArray}, x::PartialMatrix) =
    PartialMatrix(adapt(to, x.val), x.batchLength, x.batchCumSize)
Adapt.adapt_storage(to::Type{Array}, x::PartialMatrix) =
    PartialMatrix(adapt(to, x.val), x.batchLength, x.batchCumSize)

Base.size(x::PartialMatrix) = size(x.val)
function Base.getindex(x::PartialMatrix{Tv}, i::Int, j::Int) where Tv
    n = min(
        length(x.batchCumSize),
        searchsortedlast(x.batchLength, j; rev=true)
    )
    n = max(1, n)
    q = x.batchCumSize[n]
    @show i, j, q, n
    (i > q || j) > n && return zero(Tv)
    x.val[n,q]
end
