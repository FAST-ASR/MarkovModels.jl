# MarkovModels - Link (i.e. arc) of a FSM.
#
# Lucas Ondel, 2021

#######################################################################
# Concrete Link

"""
    struct Link{T}
        src::T where T<:AbstractState
        dest::D where T<:AbstractState
        weight::T
    end

Weighted link pointing from a state `src` to a state `dest` with
weight `weight`.  `T` is the type of the weight value.
The weight represents the log-probability of going through this link.
"""
struct Link{T} <: AbstractLink{T}
    src::S where S<:AbstractState
    dest::D where D<:AbstractState
    weight::T
end

