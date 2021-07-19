# SPDX-License-Identifier: MIT

# Stable implemenation of the log(exp(x) + exp(y)).
function logaddexp(x::T, y::T) where T
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
Base.promote(x::SF, y::Real) where SF <: Semifield = x, SF(y)
Base.promote(x::Real, y::SF) where SF <: Semifield = SF(x), y
Base.show(io::IO, x::Semifield) = print(io, x.val)

#======================================================================
Log-semifield:
    x ⊕ y := log( exp(x) + exp(y))
    x ⊗ y := x + y
    x ⊘ y := x - y
======================================================================#

struct LogSemifield{T<:AbstractFloat} <: Semifield
    val::T
end

Base.:+(x::LogSemifield{T}, y::LogSemifield{T}) where T =
    LogSemifield{T}(logaddexp(x.val, y.val))
Base.:*(x::LogSemifield, y::LogSemifield) = LogSemifield(x.val + y.val)
Base.:/(x::LogSemifield, y::LogSemifield) = LogSemifield(x.val - y.val)
Base.zero(::Type{LogSemifield{T}}) where T = LogSemifield{T}(T(-Inf))
Base.zero(::LogSemifield{T}) where T = LogSemifield{T}(T(-Inf))
Base.one(::Type{LogSemifield{T}}) where T = LogSemifield{T}(T(0))
Base.one(::LogSemifield{T}) where T = LogSemifield{T}(T(0))
Base.isless(x::LogSemifield, y::LogSemifield) = isless(x.val, y.val)
Base.typemin(x::Type{LogSemifield{T}}) where T = LogSemifield{T}(typemin(T))
Base.typemax(x::Type{LogSemifield{T}}) where T = LogSemifield{T}(typemax(T))

#======================================================================
Tropical-semifield:
    x ⊕ y := max(x, y)
    x ⊗ y := x + y
    x ⊘ y := x - y
======================================================================#

struct TropicalSemifield{T<:AbstractFloat} <: Semifield
    val::T
end

Base.:+(x::TropicalSemifield, y::TropicalSemifield) =
    TropicalSemifield(max(x.val, y.val))
Base.:*(x::TropicalSemifield, y::TropicalSemifield) =
    TropicalSemifield(x.val + y.val)
Base.:/(x::TropicalSemifield, y::TropicalSemifield) =
    TropicalSemifield(x.val - y.val)
Base.zero(::Type{TropicalSemifield{T}}) where T = TropicalSemifield{T}(T(-Inf))
Base.zero(::TropicalSemifield{T}) where T = TropicalSemifield{T}(T(-Inf))
Base.one(::Type{TropicalSemifield{T}}) where T = TropicalSemifield{T}(T(0))
Base.one(::TropicalSemifield{T}) where T = TropicalSemifield{T}(T(0))
Base.isless(x::TropicalSemifield, y::TropicalSemifield) = isless(x.val, y.val)
Base.typemin(x::Type{TropicalSemifield{T}}) where T =
    TropicalSemifield{T}(typemin(T))
Base.typemax(x::Type{TropicalSemifield{T}}) where T =
    TropicalSemifield{T}(typemax(T))

