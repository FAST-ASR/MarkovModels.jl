import StatsFuns: logsumexp

function logsumexp(X::AbstractArray{T}; dims=:) where {T<:Real}
    u = reduce(max, X, dims=dims, init=oftype(log(zero(T)), -Inf))
    u isa AbstractArray || isfinite(u) || return float(u)
    let u=u
        if u isa AbstractArray
            v = u .+ log.(sum(exp.(X .- u); dims=dims))
            i = .! isfinite.(v)
            v[i] .= u[i]
            v
        else
            u + log(sum(x -> exp(x-u), X))
        end
    end
end