# MarkovModels - Link (i.e. arc) of a FSM.
#
# Lucas Ondel, 2021

#######################################################################
# Link

"""
    struct Link{T}
        src::T where T<:AbstractState
        dest::D where T<:AbstractState
        weight::T
    end

Weighted link pointing to a destination state `dest` with
weight `weight`.
"""
struct Link{S<:AbstractState,T<:SemiField}
    dest::S
    weight::T
end

