#A = CuSparseMatrixCSR(Aᵀ.colPtr, Aᵀ.rowVal, Aᵀ.nzVal, A.dims) SPDX-License-Identifier: MIT

const kMinLogDiffFloat = log(1.19209290f-7)

function mylogaddexp(x::T, y::T) where T
    if x == T(-Inf) return y end
    if y == T(-Inf) return x end

    diff = T(0)
    if x < y
        diff = x - y
        x = y
    else
        diff = y -x
    end

    if diff >= log(eps(T))
        return x + log1p(exp(diff))
    end
    return x
end

abstract type Semifield <: Number end

Base.convert(T::Type{<:Real}, x::Semifield) = T(x.val)

struct LogSemifield{T<:AbstractFloat} <: Semifield
    val::T
end

Base.show(io::IO, x::LogSemifield) = print(io, x.val)
Base.promote(x::LogSemifield{T}, y::Real) where T = x, LogSemifield{T}(y)
Base.promote(y::Real, x::LogSemifield{T}) where T = LogSemifield{T}(y), x
Base.:+(x::LogSemifield{T}, y::LogSemifield{T}) where T =
    LogSemifield{T}(logaddexp(x.val, y.val))
Base.:*(x::LogSemifield, y::LogSemifield) = LogSemifield(x.val + y.val)
Base.:/(x::LogSemifield, y::LogSemifield) = LogSemifield(x.val - y.val)
Base.zero(::Type{LogSemifield{T}}) where T = LogSemifield{T}(T(-Inf))
Base.zero(::LogSemifield{T}) where T = LogSemifield{T}(T(-Inf))
Base.one(::Type{LogSemifield{T}}) where T = LogSemifield{T}(T(0))
Base.one(::LogSemifield{T}) where T = LogSemifield{T}(T(0))
Base.isless(x::LogSemifield, y::LogSemifield) = isless(x.val, y.val)
Base.abs(x::LogSemifield) = abs(x.val)
Base.typemin(x::Type{LogSemifield{T}}) where T = LogSemifield{T}(typemin(T))
Base.typemax(x::Type{LogSemifield{T}}) where T = LogSemifield{T}(typemax(T))

struct RealSemifield{T<:AbstractFloat} <: Semifield
    val::T
end

export RealSemifield

Base.show(io::IO, x::RealSemifield) = print(io, x.val)
Base.:+(x::RealSemifield, y::RealSemifield) = RealSemifield(x.val + y.val)
Base.:*(x::RealSemifield, y::RealSemifield) = RealSemifield(x.val + y.val)
Base.:/(x::RealSemifield, y::RealSemifield) = RealSemifield(x.val - y.val)
Base.zero(::Type{RealSemifield{T}}) where T = RealSemifield{T}(T(-Inf))
Base.zero(::RealSemifield{T}) where T = RealSemifield{T}(T(-Inf))
Base.one(::Type{RealSemifield{T}}) where T = RealSemifield{T}(T(0))
Base.one(::RealSemifield{T}) where T = RealSemifield{T}(T(0))
Base.isless(x::RealSemifield, y::RealSemifield) = isless(x.val, y.val)
Base.abs(x::RealSemifield) = abs(x.val)
Base.typemin(x::Type{RealSemifield{T}}) where T = RealSemifield{T}(typemin(T))
Base.typemax(x::Type{RealSemifield{T}}) where T = RealSemifield{T}(typemax(T))
