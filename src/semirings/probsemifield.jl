# SPDX-License-Identifier: MIT

#======================================================================
Probability-semifield:
    x ⊕ y := x + y
    x ⊗ y := x * y
    x ⊘ y := x / y
======================================================================#

struct ProbabilitySemifield{T<:AbstractFloat} <: Semifield
    val::T
end

Base.:+(x::ProbabilitySemifield{T}, y::ProbabilitySemifield{T}) where T =
    ProbabilitySemifield{T}(x.val + y.val)
Base.:*(x::ProbabilitySemifield, y::ProbabilitySemifield) =
    ProbabilitySemifield(x.val * y.val)
Base.:/(x::ProbabilitySemifield, y::ProbabilitySemifield) =
    ProbabilitySemifield(x.val / y.val)
Base.zero(::Type{ProbabilitySemifield{T}}) where T =
    ProbabilitySemifield{T}(T(0))
Base.one(::Type{ProbabilitySemifield{T}}) where T =
    ProbabilitySemifield{T}(T(1))
Base.isless(x::ProbabilitySemifield, y::ProbabilitySemifield) =
    isless(x.val, y.val)
Base.typemin(x::Type{ProbabilitySemifield{T}}) where T = zero(T)
Base.typemax(x::Type{ProbabilitySemifield{T}}) where T = one(T)
