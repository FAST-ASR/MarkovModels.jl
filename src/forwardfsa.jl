# SPDX-License-Identifier: MIT

#======================================================================
Dense forward FSA
======================================================================#


struct BlockForwardInitVector{K, Tv<:AbstractVector{K}} <: AbstractVector{K}
    data::Tv
    nblocks::Int64
end

Base.size(v::BlockForwardInitVector) = (length(v.data) * v.nblocks,)

Base.getindex(v::BlockForwardInitVector{K}, i::Int) where K =
    i <= length(v.data) ? v.data[i] : zero(K)


struct BlockForwardFinalVector{K, Tv<:AbstractVector{K}} <: AbstractVector{K}
    data::Tv
    nblocks::Int64
end

Base.size(v::BlockForwardFinalVector) = (length(v.data) * v.nblocks,)

function Base.getindex(v::BlockForwardFinalVector{K}, i::Int) where K
    blocksize = length(v.data)
    idx = (i - 1) % blocksize + 1
    i <= (v.nblocks - 1) * length(v.data) ? zero(K) : v.data[idx]
end


struct BlockForwardMatrix{K, TM<:AbstractMatrix{K}} <: AbstractMatrix{K}
    data::TM
end

function Base.size(M::BlockForwardMatrix)
    dim = size(M.data, 1) * (size(M.data, 2) + 1)
    (dim, dim)
end

function Base.getindex(M::BlockForwardMatrix{K}, i::Int, j::Int)  where K
    blocksize = size(M.data, 1)
    rowblock = (i - 1) ÷ blocksize + 1
    colblock = (j - 1) ÷ blocksize + 1
    if colblock == rowblock + 1
        return M.data[(i - 1) % blocksize + 1, colblock - 1]
    end
    zero(K)
end

function DenseForwardFSA(M::AbstractMatrix{K}, λ = nothing) where K
    if isnothing(λ)
        λ = DefaultSymbolTable(prod(size(M)))
    end
    FSA(
        BlockForwardInitVector(view(M, :, 1), size(M, 2)),
        BlockForwardMatrix(view(M, :, 2:size(M, 2))),
        BlockForwardFinalVector(
            fill!(similar(M, size(M, 1)), one(K)),
            size(M, 2)
        ),
        λ
    )
end
