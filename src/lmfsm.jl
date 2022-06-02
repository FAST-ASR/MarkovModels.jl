# SPDX-License-Identifier: MIT

"""
    totalngramsum(fsm::FSM; order)

Calculate the n-gram statistics from `fsm` of order `order`.
"""
function totalngramsum(fsm::FSM; order)
    K = eltype(fsm.α)

    # To avoid missing sequences shorter than `order` with "pad"
    # FSM with empty states.
    if order > 1
        pad = FSM(
              sparsevec([1], [one(K)], order-1),
              spdiagm(1 => ones(K, order-2)),
              sparsevec([order-1], [one(K)], order-1),
              repeat([Label()], order-1)
        )
        fsm = cat(pad, fsm)
    end

    T1 = ProductSemiring{AppendConcatSemiring{LabelMonoid},K}
    T2 = ProductSemiring{K,K}
    S = ProductSemiring{T1,T2}

    α = [AppendConcatSemiring([S(T1(AppendConcatSemiring([λᵢ]), one(K)),
                                    T2(αᵢ, one(K)))])
         for (λᵢ, αᵢ) in zip(fsm.λ, fsm.α)]

    I, J, V = findnz(fsm.T)
    V2 = [AppendConcatSemiring([S(T1(AppendConcatSemiring([fsm.λ[J[i]]]),
                                        V[i]), one(T2))])
          for i in 1:length(V)]
    T = sparse(I, J, V2, nstates(fsm), nstates(fsm))

    ω = [AppendConcatSemiring([S(one(T1), T2(one(K), ωᵢ))])
         for ωᵢ in fsm.ω]

    # N-gram statistics are evaluated with the total-sum algorithm.
    stats = totalsum(α, T, ω, order)

    # We merge all identical n-grams
    ngrams = Dict()
    for s in val(stats)
        ((array_ngram, w), (iw, fw)) = _unfold(s)

        # 1. By construction, `array_ngram` is guaranteed to have only
        #    one element.
        # 2. We need to reverse the sequence as Julia does not respect
        #    the right or left multiplication (it always
        #    right-multiply).
        ngram = reverse(val(val(array_ngram)[1]))

        a, b, c = get(ngrams, ngram, (zero(K), zero(K), zero(K)))
        ngrams[ngram] = (a+iw, b+w, c+fw)
    end

    ngrams
end

"""
    LanguagageModelFSM(ngrams)

Build a language model FSM from ngram statistics.
"""
function LanguageModelFSM(ngrams)
	states = Dict()
	initstates = Dict()
	finalstates = Dict()
	arcs = Dict()

    # Calcultate the ngram order by checking the maximum ngram length.
    order = 0
    for key in keys(ngrams)
        order = max(length(key), order)
    end

	for (ngram, (iw, w, fw)) in ngrams
		if length(ngram) == 1 && ! iszero(iw)
			states[ngram] = get(states, ngram, length(states) + 1)
			initstates[ngram] = iw + get(initstates, ngram, zero(iw))
			if ! iszero(fw)
				finalstates[ngram] = fw + get(finalstates, ngram, zero(fw))
			end
		elseif length(ngram) > 1
			src = ngram[1:min(order, length(ngram)) - 1]
			dest = ngram[max(1, length(ngram) - order + 2):end]
			states[src] = get(states, src, length(states) + 1)
			states[dest] = get(states, dest, length(states) + 1)
			arcs[(src, dest)] = w + get(arcs, (src, dest), zero(w))

			if ! iszero(fw)
				finalstates[dest] = fw + get(finalstates, dest, zero(fw))
			end
		end
	end

	FSM(
		[states[s] => v for (s, v) in initstates],
		[(states[src], states[dest]) => v for ((src, dest), v) in arcs],
		[states[s] => v for (s, v) in finalstates],
		[SequenceMonoid(s) for (s, _) in sort(collect(states), by = p -> p[2])]
	) |> renorm
end

