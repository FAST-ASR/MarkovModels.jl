# MarkovModels.jl
#
# Lucas Ondel, 2021

mutable struct State{T<:SemiField}
    id::UInt64
    startweight::T
    finalweight::T
end

#Base.:(==)(s1::State, s2::State) = s1.id == s2.id
#Base.hash(s::State, h::UInt) = hash(s.id, h)

isinit(s::State{T}) where T = s.startweight ≠ zero(T)
isfinal(s::State{T}) where T = s.finalweight ≠ zero(T)

setstart!(s::State{T}, weight::T = one(T)) where T = s.startweight = weight
setfinal!(s::State{T}, weight::T = one(T)) where T = s.finalweight = weight

