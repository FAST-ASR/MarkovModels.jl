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

abstract type Semiring <: Number end
abstract type Semifield <: Semiring end

Base.zero(x::T) where T<:Semiring = zero(T)
Base.convert(::Type{T}, x::T) where T<:Semiring = x
Base.convert(T::Type{<:Number}, x::Semiring) = T(x.val)
Base.promote_rule(x::Type{T}, y::Type{<:Real}) where T <: Semifield = T
Base.show(io::IO, x::Semiring) = print(io, x.val)

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

#======================================================================
Tropical-semifield:
    x ⊕ y := max(x, y)
    x ⊗ y := x + y
======================================================================#

struct TropicalSemiring{T<:AbstractFloat} <: Semiring
    val::T
end

Base.:+(x::TropicalSemiring, y::TropicalSemiring) =
    TropicalSemiring(max(x.val, y.val))
Base.:*(x::TropicalSemiring, y::TropicalSemiring) =
    TropicalSemiring(x.val + y.val)
Base.zero(::Type{TropicalSemiring{T}}) where T = TropicalSemiring{T}(T(-Inf))
Base.one(::Type{TropicalSemiring{T}}) where T = TropicalSemiring{T}(T(0))
Base.isless(x::TropicalSemiring, y::TropicalSemiring) = isless(x.val, y.val)
Base.typemin(x::Type{TropicalSemiring{T}}) where T =
    TropicalSemiring{T}(typemin(T))
Base.typemax(x::Type{TropicalSemiring{T}}) where T =
    TropicalSemiring{T}(typemax(T))

#======================================================================
Viterbi-semiring:
    x ⊕ y := max(x, y)
    x ⊗ y := x + y
======================================================================#

struct ViterbiSemiring{Tv<:AbstractFloat,Ts} <: Semiring
    val::Tv
    seq::Tuple{Vararg{Tv}}
end

function Base.:+(x::ViterbiSemiring, y::ViterbiSemiring)
    tup = x.val ≥ y.val ? (x.val, x.seq) : (y.val, y.seq)
    ViterbiSemiring(tup...)
end

Base.:*(x::ViterbiSemiring, y::ViterbiSemiring) =
    ViterbiSemiring(x.val + y.val, tuple(x.seq..., y.seq...))
Base.zero(::Type{ViterbiSemiring{Tv,Ts}}) where {Tv,Ts} =
    ViterbiSemiring{Tv,Ts}(-Inf, tuple())
Base.one(::Type{ViterbiSemiring{Tv,Ts}}) where {Tv,Ts} =
    ViterbiSemiring{Tv,Ts}(0, tuple())
Base.isless(x::ViterbiSemiring, y::ViterbiSemiring) = isless(x.val, y.val)
