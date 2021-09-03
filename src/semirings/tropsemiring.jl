# SPDX-License-Identifier: MIT

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
