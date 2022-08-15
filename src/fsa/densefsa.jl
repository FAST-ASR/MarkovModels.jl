# SPDX-License-Identifier: MIT

#const DenseFSA{K} = FSA{K, Matrix{K}} where K

fsatype(fsa::FSA{K, T}) where {K, T<:Matrix{K}} = "dense"

function DenseFSA(initws, arcs, finalws, λ; kwargs...)
    # Get the semiring of the FSA.
    K = typeof(initws[1][2])

    # Get the set of states indices.
    states = reduce(
        union,
        [
            Set(map(first, initws)),
            Set(filter(x -> x > 0, map(x -> x.first[1], arcs))),
            Set(filter(x -> x > 0, map(x -> x.first[2], arcs))),
            Set(map(first, finalws))
        ]
    )
    nstates = length(states)

    α = zeros(K, nstates)
    for (i, w) in initws
        α[i] = w
    end

    T = zeros(K, nstates, nstates)
    for ((i, j), w) in arcs
        T[i, j]  = w
    end

    ω = zeros(K, nstates)
    for (i, w) in finalws
        ω[i] = w
    end

    FSA(α, T, ω, λ; kwargs...)
end
