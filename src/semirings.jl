# SPDX-License-Identifier: MIT

"""
    abstract type Semiring <: Number end

Abstract base type for all semirings. All concrete type of `Semiring`
should have a `val` attribute.
"""
abstract type Semiring <: Number end

Base.zero(x::T) where T<:Semiring = zero(T)
Base.one(x::T) where T<:Semiring = one(T)
Base.convert(T::Type{<:Number}, x::Semiring) = T(x.val)
Base.conj(x::Semiring) = conj(x.val)
Base.promote_rule(x::Type{Semiring}, y::Type{Number}) = Semiring
Base.show(io::IO, x::Semiring) = print(io, x.val)

#======================================================================
Semiring properties
======================================================================#

"""
    abstract type SemiringProperty end

Abstract base type for all semiring properties.
"""
abstract type SemiringProperty end

abstract type SemiringDivisibleProperty <: SemiringProperty end
struct IsDivisible <: SemiringDivisibleProperty end
struct IsNotDivisible <: SemiringDivisibleProperty end
SemiringDivisibleProperty(::Type) = IsNotDivisible()

abstract type SemiringOrderedProperty <: SemiringProperty end
struct IsOrdered <: SemiringOrderedProperty end
struct IsNotOrdered <: SemiringDivisibleProperty end
SemiringOrderedProperty(::Type) = IsNotOrdered()

#======================================================================
alternation-concatenation-semiring
======================================================================#

const SymbolSequence{N} = NTuple{N, Any} where N
SymbolSequence(syms) = SymbolSequence{length(syms)}(syms)

"""
    struct ACSemiring <: Semiring
        val::Set{<:SymbolSequence}
    end

Alternation-Concatenation semiring.
"""
struct ACSemiring <: Semiring
    val::Set{<:SymbolSequence}
end

Base.:+(x::ACSemiring, y::ACSemiring) = ACSemiring(union(x.val, y.val))
function Base.:*(x::ACSemiring, y::ACSemiring)
    newseqs = Set{SymbolSequence}()
    for xᵢ in x.val
        for yᵢ in y.val
            push!(newseqs, SymbolSequence(vcat(xᵢ..., yᵢ...)))
        end
    end
    ACSemiring(newseqs)
end

Base.zero(::ACSemiring) = ACSemiring(Set{SymbolSequence}())
Base.one(::ACSemiring) = ACSemiring(Set([tuple()]))
Base.conj(x::ACSemiring) = identity

#======================================================================
log-semiring
======================================================================#

# Stable implementation of the log(exp(x) + exp(y)).
function logaddexp(x, y)
    diff = zero(promote_type(typeof(x), typeof(y)))
    if x < y
        diff = x - y
        x = y
    else y < x
        diff = y - x
    end

    if diff >= log(eps())
        return x + log1p(exp(diff))
    else
        return x
    end
end

"""
    struct LogSemiring{T<:AbstractFloat} <: Semfield
        val::T
    end

The log semiring is defined as :
  * ``x + y \\triangleq \\ln( e^x + e^y)``
  * ``x \times y \\triangleq x + y``
  * ``x^{-1} \\triangleq -x``
``\\forall x, y \\in \\mathbb{R}``.
"""
struct LogSemiring{T<:AbstractFloat} <: Semiring
    val::T
end

SemiringOrderedProperty(::Type{<:LogSemiring}) = IsOrdered()
SemiringDivisibleProperty(::Type{<:LogSemiring}) = IsDivisible()

Base.:+(x::LogSemiring, y::LogSemiring) = LogSemiring(logaddexp(x.val, y.val))
Base.:*(x::LogSemiring, y::LogSemiring) = LogSemiring(x.val + y.val)
Base.:/(x::LogSemiring, y::LogSemiring) = LogSemiring(x.val - y.val)
Base.zero(::Type{LogSemiring{T}}) where T = LogSemiring(T(-Inf))
Base.one(::Type{LogSemiring{T}}) where T = LogSemiring(T(0))
Base.isless(x::LogSemiring, y::LogSemiring) = isless(x.val, y.val)
Base.typemin(x::Type{LogSemiring{T}}) where T = LogSemiring{T}(typemin(T))
Base.typemax(x::Type{LogSemiring{T}}) where T = LogSemiring{T}(typemax(T))

#======================================================================
prob-semiring
======================================================================#

"""
    struct ProbabilitySemiring{T<:AbstractFloat} <: Semiring
        val::T
    end

Probability semiring defined as :
  * ``x \\oplus y \\triangleq x + y``
  * ``x \\otimes y \\triangleq x \\cdot y``
  * ``x \\oslash y \\triangleq \\frac{x}{y}``
``\\forall x, y \\in [0, 1]``.
"""
struct ProbabilitySemiring{T<:AbstractFloat} <: Semiring
    val::T
end

SemiringOrderedProperty(::Type{<:LogSemiring}) = IsOrdered()
SemiringDivisibleProperty(::Type{<:LogSemiring}) = IsDivisible()

Base.:+(x::ProbabilitySemiring, y::ProbabilitySemiring) =
    ProbabilitySemiring(x.val + y.val)
Base.:*(x::ProbabilitySemiring, y::ProbabilitySemiring) =
    ProbabilitySemiring(x.val * y.val)
Base.:/(x::ProbabilitySemiring, y::ProbabilitySemiring) =
    ProbabilitySemifield(x.val / y.val)
Base.zero(::Type{ProbabilitySemifield{T}}) where T = ProbabilitySemiring(T(0))
Base.one(::Type{ProbabilitySemifield{T}}) where T = ProbabilitySemiring(T(1))
Base.isless(x::ProbabilitySemiring, y::ProbabilitySemiring) = isless(x.val, y.val)
Base.typemin(x::Type{ProbabilitySemifield{T}}) where T = zero(T)
Base.typemax(x::Type{ProbabilitySemifield{T}}) where T = one(T)

