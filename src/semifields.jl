# SPDX-License-Identifier: MIT

#######################################################################
# The following functions are taken from the LogExpFunctions package:
#   https://github.com/JuliaStats/LogExpFunctions.jl

function logaddexp(x::Real, y::Real)
    # ensure Δ = 0 if x = y = ± Inf
    Δ = ifelse(x == y, zero(x - y), abs(x - y))
    max(x, y) + log1pexp(-Δ)
end

log1pexp(x::Real) = x < 18.0 ? log1p(exp(x)) : x < 33.3 ? x + exp(-x) : oftype(exp(-x), x)
log1pexp(x::Float32) = x < 9.0f0 ? log1p(exp(x)) : x < 16.0f0 ? x + exp(-x) : oftype(exp(-x), x)

#######################################################################

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
