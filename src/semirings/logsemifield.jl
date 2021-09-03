# SPDX-License-Identifier: MIT

# Stable implementation of the log(exp(x) + exp(y)).
function logaddexp(x::T, y::T) where T
    diff = zero(T)
    if x < y
        diff = x - y
        x = y
    else y < x
        diff = y - x
    end

    if diff >= log(eps(T))
        return x + log1p(exp(diff))
    end
    return x
end

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
Base.one(::Type{LogSemifield{T}}) where T = LogSemifield{T}(T(0))
Base.isless(x::LogSemifield, y::LogSemifield) = isless(x.val, y.val)
Base.typemin(x::Type{LogSemifield{T}}) where T = LogSemifield{T}(typemin(T))
Base.typemax(x::Type{LogSemifield{T}}) where T = LogSemifield{T}(typemax(T))