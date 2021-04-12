# Lucas Ondel, 2021

const LogSemifield{T} = Semifield{T, logaddexp, +, -, -Inf, 0} where T
upperbound(::Type{<:T}) where T<:LogSemifield = T(Inf)
lowerbound(::Type{<:T}) where T<:LogSemifield = T(-Inf)

const TropicalSemifield{T} = Semifield{T, max, +, -, -Inf, 0} where T
upperbound(::Type{<:T}) where T<:TropicalSemifield = T(Inf)
lowerbound(::Type{<:T}) where T<:TropicalSemifield = T(-Inf)

Base.convert(T::Type{<:Real}, x::Semifield) = T(x.val)

# We used ordered semifield (necessary for pruning).
const OrderedSemifield = Union{LogSemifield, TropicalSemifield}

Base.isless(x::OrderedSemifield, y::OrderedSemifield) = isless(x.val, y.val)
Base.isless(x::OrderedSemifield, y::Number) = isless(x.val, y)
Base.isless(x::Number, y::OrderedSemifield) = isless(x, y.val)
Base.abs(x::OrderedSemifield) = abs(x.val)

