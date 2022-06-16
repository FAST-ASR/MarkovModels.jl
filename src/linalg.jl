# SPDX-License-Identifier: MIT

struct CuSparseVecStyle <: Broadcast.AbstractArrayStyle{1} end
Broadcast.BroadcastStyle(::Type{<:CuSparseVector}) = CuSparseVecStyle()

const CuSPVM = CuSparseVecStyle

CuSparseVecStyle(::Val{0}) = CuSparseVecStyle()
CuSparseVecStyle(::Val{1}) = CuSparseVecStyle()

function Base.copy(bc::Broadcasted{<:CuSPVM})
    bcf = flatten(bc)
    #return bc, bcf
    _copy(bcf.f, bcf.args...)
end

function Base.copyto!(dest::CuSparseVector, bc::Broadcasted{<:CuSPVM})
    bcf = flatten(bc)
    dest
end

function _copy(::typeof(*), x::CuSparseVector, y::CuSparseVector)
    println("broadcasting cusparsevector")
    x
end
