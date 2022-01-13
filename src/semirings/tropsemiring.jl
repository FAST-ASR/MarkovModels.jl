# SPDX-License-Identifier: MIT


"""
    struct TropicalSemiring{T<:AbstractFloat} <: Semiring
        val::T
    end

Log-semifield is defined as :
  * ``x \\oplus y \\triangleq \\text{max}(x, y)``
  * ``x \\otimes y \\triangleq x + y``
``\\forall x, y \\in \\mathbb{R}``.
"""
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
