# SPDX-License-Identifier: MIT

function SparseFSA(initws, arcs, finalws, λ = missing)
    # Get the semiring of the FSM.
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

    # Count the epsilon states.
    eps_states = reduce(
        union,
        [
            Set(filter(x -> x <= 0, map(x -> x.first[1], arcs))),
            Set(filter(x -> x <= 0, map(x -> x.first[2], arcs))),
        ]
    )
    n_eps = length(eps_states)

    if n_eps > 0
        T = _build_T_with_epsilon(arcs, nstates, n_eps, K)
    else
        T = _build_T_no_epsilon(arcs, nstates, K)
    end

    FSA(
        sparsevec(map(x -> x[1], initws), map(x -> x[2], initws), nstates),
        T,
        sparsevec(map(x -> x[1], finalws), map(x -> x[2], finalws), nstates),
        ismissing(λ) ? DefaultSymbolTable(nstates) : λ
    )
end
