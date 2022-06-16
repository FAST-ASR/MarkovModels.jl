# SPDX-License-Identifier: MIT

struct MyCuSparseVecStyle <: Broadcast.AbstractArrayStyle{1} end
#struct CuVecStyle <: Broadcast.AbstractArrayStyle{1} end
Broadcast.BroadcastStyle(::Type{<:CuSparseVector}) = MyCuSparseVecStyle()
#Broadcast.BroadcastStyle(::Type{<:CuVector}) = CuVecStyle()

const CuSPVM = MyCuSparseVecStyle

MyCuSparseVecStyle(::Val{0}) = MyCuSparseVecStyle()
MyCuSparseVecStyle(::Val{1}) = MyCuSparseVecStyle()
#CuVecStyle(::Val{0}) = CuVecStyle()
#CuVecStyle(::Val{1}) = CuVecStyle()

Broadcast.BroadcastStyle(::MyCuSparseVecStyle, ::CUDA.CuArrayStyle) =
    MyCuSparseVecStyle()
Broadcast.BroadcastStyle(::MyCuSparseVecStyle, ::CUDA.CUSPARSE.CuSparseVecStyle) =
    MyCuSparseVecStyle()

function Base.copy(bc::Broadcasted{<:CuSPVM})
    println("my copy")
    bcf = flatten(bc)
    #return bc, bcf
    _copy(bcf.f, bcf.args...)
end

function Base.copyto!(dest::CuSparseVector, bc::Broadcasted{<:CuSPVM})
    println("my copyto!")
    bcf = flatten(bc)
    _copyto!(bcf.f, dest, bcf.args...)
end

_copyto!(::typeof(*), out::CuSparseVector, x::CuSparseVector, y::CuVector) =
    elmul_svdv!(out, x, y)
_copy(f, x, y) = _copyto!(f, similar(x), y)

