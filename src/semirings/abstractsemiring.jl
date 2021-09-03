# SPDX-License-Identifier: MIT

abstract type Semiring <: Number end
abstract type Semifield <: Semiring end

Base.zero(x::T) where T<:Semiring = zero(T)
Base.convert(::Type{T}, x::T) where T<:Semiring = x
Base.convert(T::Type{<:Number}, x::Semiring) = T(x.val)
Base.promote_rule(x::Type{T}, y::Type{<:Real}) where T <: Semiring = T
Base.show(io::IO, x::Semiring) = print(io, x.val)
