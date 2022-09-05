struct FactorialFSM{K<:Semiring}
    fsms::Vector{FSM{K}}
    smaps::Vector{AbstractMatrix{K}}
end

function FactorialFSM(
            fsm1::FSM{K}, smap1::AbstractMatrix{K},
            fsm2::FSM{K}, smap2::AbstractMatrix{K}
) where K 
    FactorialFSM([fsm1, fsm2], [smap1, smap2])
end

function init_messages(ffsm::FactorialFSM{K}, N::Integer) where K
    dims = [nstates(fsm) for fsm in ffsm.fsms]
    nspkrs = length(dims)

    m1 = [Array{K}(undef, dims[j], N) for j in nspkrs]
    m2 = [Array{K}(undef, dims[j], N) for j in nspkrs]
    m3 = [Array{K}(undef, dims[j], N) for j in nspkrs]

    @views for j in 1:nspkrs
        S = dims(j) # n_states in fsm for spkr j
        m1j, m2j, m3j = m1[j], m2[j], m3[j] 
        fill!(m2[:, 2:end], one(K) / S)
        m2[1, :] = 
    end

    (m1 = m1, m2 = m2, m3 = m3)
end

function lbp_posteriors(ffsm::FactorialFSM{K}, llhs::AbstractArray{K, J}; eps=1e-4) where {K, J}
    messages = init_messages(ffsm, llhs)

    while true
        new_messages = lbp_step(messages)
        # check the difference between messages
        diffs = [.≈(new_m, m; atol=eps) for (new_m, m) in zip(new_messages, messages)]
        # if all messages are same then break
        all(all.(diffs)) && break
        old_messages = new_messages
    end
end

function lbp_step(messages)

end
