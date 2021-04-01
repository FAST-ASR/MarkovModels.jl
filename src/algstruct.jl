# MarkovModels.jl
#
# Algebraic structures
#
# Lucas Ondel, 2021

"""
    SemiField{T,A,M,MI,Z,O}

`T` is the value type, `A` is the addition function, `M` is the
multiplication function, `MI` is the muliplication inverse function,
`Z` is the zero element and `O` is the one element.
"""
struct SemiField{T,A,M,MI,Z,O} <: Number
    val::T
end

Base.show(io::IO, x::SemiField) = print(io, x.val)

Base.isless(x::SemiField, y::SemiField) = isless(x.val, y.val)
Base.isless(x::SemiField, y::Number) = isless(x.val, y)
Base.isless(x::Number, y::SemiField) = isless(x, y.val)
Base.abs(x::SemiField) = abs(x.val)

Base.:+(x::SemiField{T,A,M,MI,Z,O}, y::SemiField{T,A,M,MI,Z,O}) where {T,A,M,MI,Z,O} =
    SemiField{T,A,M,MI,Z,O}( A(x.val, y.val) )
Base.:*(x::SemiField{T,A,M,MI,Z,O}, y::SemiField{T,A,M,MI,Z,O}) where {T,A,M,MI,Z,O} =
    SemiField{T,A,M,MI,Z,O}( M(x.val, y.val) )
Base.:/(x::SemiField{T,A,M,MI,Z,O}, y::SemiField{T,A,M,MI,Z,O}) where {T,A,M,MI,Z,O} =
    SemiField{T,A,M,MI,Z,O}( MI(x.val, y.val) )
Base.zero(::Type{SemiField{T,A,M,MI,Z,O}}) where {T,A,M,MI,Z,O} = SemiField{T,A,M,MI,Z,O}(Z)
Base.zero(::SemiField{T,A,M,MI,Z,O}) where {T,A,M,MI,Z,O} = SemiField{T,A,M,MI,Z,O}(Z)
Base.one(::Type{SemiField{T,A,M,MI,Z,O}})  where {T,A,M,MI,Z,O} = SemiField{T,A,M,MI,Z,O}(O)
Base.one(::SemiField{T,A,M,MI,Z,O}) where {T,A,M,MI,Z,O} = SemiField{T,A,M,MI,Z,O}(O)

const LogSemiField{T} = SemiField{T, logaddexp, +, -, -Inf, 0} where T
const MaxTropicalSemiField{T} = SemiField{T, max, +, -, -Inf, 0} where T
const MinTropicalSemiField{T} = SemiField{T, min, +, -, Inf, 0} where T
const ProbabilitySemiField{T} = SemiField{T, +, *, /, 0, 1} where T

upperbound(::Type{<:T}) where T<:LogSemiField = T(Inf)
lowerbound(::Type{<:T}) where T<:LogSemiField = T(-Inf)
upperbound(::Type{<:T}) where T<:MaxTropicalSemiField = T(Inf)
lowerbound(::Type{<:T}) where T<:MaxTropicalSemiField = T(-Inf)
upperbound(::Type{<:T}) where T<:ProbabilitySemiField = T(Inf)
lowerbound(::Type{<:T}) where T<:ProbabilitySemiField = T(0)

Base.convert(T::Type{<:Real}, x::ProbabilitySemiField) = T(x.val)
Base.convert(T::Type{<:Real}, x::LogSemiField) = T(exp(x.val))
Base.convert(T::Type{<:Real}, x::MaxTropicalSemiField) = T(exp(x.val))
Base.convert(T::Type{<:LogSemiField}, x::MaxTropicalSemiField) = T(x.val)
Base.convert(T::Type{<:LogSemiField}, x::ProbabilitySemiField) = T(log(x.val))
Base.convert(T::Type{<:LogSemiField}, x::Real) = T(log(x))
Base.convert(T::Type{<:MaxTropicalSemiField}, x::LogSemiField) = T(x.val)
Base.convert(T::Type{<:MaxTropicalSemiField}, x::ProbabilitySemiField) = T(log(x.val))
Base.convert(T::Type{<:MaxTropicalSemiField}, x::Real) = T(log(x))
Base.convert(T::Type{<:ProbabilitySemiField}, x::LogSemiField) = T(exp(x.val))
Base.convert(T::Type{<:ProbabilitySemiField}, x::MaxTropicalSemiField) = T(exp(x.val))
Base.convert(T::Type{<:ProbabilitySemiField}, x::Real) = T(x)

