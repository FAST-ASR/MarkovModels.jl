# MarkovModels.jl
#
# Lucas Ondel, 2021

const Label = Union{String,Nothing}

struct Link{T<:SemiField}
    dest::State
    ilabel::Label
    olabel::Label
    weight::T
end

