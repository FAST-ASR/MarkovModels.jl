# SPDX-License-Identifier: MIT

abstract type Semifield <: Number end

struct LogSemifield{T<:AbstractFloat} <: Semifield
    val::T
end

Base.show(io::IO, x::LogSemifield) = print(io, x.val)
Base.:+(x::LogSemifield, y::LogSemifield) = LogSemifield(logaddexp(x.val, y.val))
Base.:*(x::LogSemifield, y::LogSemifield) = LogSemifield(x.val + y.val)
Base.:/(x::LogSemifield, y::LogSemifield) = LogSemifield(x.val - y.val)
Base.zero(::Type{LogSemifield{T}}) where T = LogSemifield{T}(T(-Inf))
Base.zero(::LogSemifield{T}) where T = LogSemifield{T}(T(-Inf))
Base.one(::Type{LogSemifield{T}}) where T = LogSemifield{T}(T(0))
Base.one(::LogSemifield{T}) where T = LogSemifield{T}(T(0))
Base.convert(T::Type{<:Real}, x::Semifield) = T(x.val)
Base.isless(x::LogSemifield, y::LogSemifield) = isless(x.val, y.val)
Base.abs(x::LogSemifield) = abs(x.val)
Base.typemin(x::Type{LogSemifield{T}}) where T = LogSemifield{T}(typemin(T))
Base.typemax(x::Type{LogSemifield{T}}) where T = LogSemifield{T}(typemax(T))
