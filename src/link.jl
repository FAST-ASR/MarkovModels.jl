# MarkovModels.jl
#
# Lucas Ondel, 2021

"""
    struct Link{S,L,T}
        dest::S
        label::L
        weight::T
    end

Weighted link pointing to a destination state `dest` with
weight `weight`.
"""
struct Link{S<:AbstractState,L<:AbstractString,T:SemiField}
    dest::S
    label::L
    weight::T
end

