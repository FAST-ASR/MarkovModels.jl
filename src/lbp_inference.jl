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
        fsm = ffsm.fsms[j]
        S = nstates(fsm)
        m1j, m2j, m3j = m1[j], m2[j], m3[j] 
        fill!(m2j[:, 2:end], one(K))
        fill!(m2j[:, 1], fsm.α)
        fill!(m3j, one(K))
    end

    (m1 = m1, m2 = m2, m3 = m3)
end

function lbp_posteriors(ffsm::FactorialFSM{K}, llhs::AbstractArray{K, J}; eps=1e-4) where {K, J}
    N = size(llhs, ndims(llhs))
    messages = init_messages(ffsm, N)

    while true
        new_messages = lbp_step!(deepcopy(messages), ffsm, llhs)
        # check the difference between messages
        diffs = [.≈(new_m, m; atol=eps) for (new_m, m) in zip(new_messages, messages)]
        # if all messages are same then break
        all(all.(diffs)) && break
        messages = new_messages
    end

    m1, m2, m3 = messages
end

function lbp_step!(messages, ffsm::FactorialFSM{K}, llhs::AbstractArray{K, 3}) where K
    n_spkrs = ndims(llhs) - 1
    @assert n_spkrs == 2 "Currently we do not support more than 2 speakers!"

    m1, m2, m3 = messages
    N = size(llhs, ndims(llhs))

    for j in 1:n_spkrs
        # this spkr's messages
        m1j, m2j, m3j = m1[j], m2[j], m3[j]
        fsm = ffsm.fsms[j]
        T̂, T̂ᵀ = fsm.T̂, permutedims(fsm.T̂, [2,1]) # TODO maybe not optimal

        # other spkr's messages
        k = n_spkrs - j + 1
        m2k, m3k = m2[k], m3[k]
        buffer_k = similar(m2k[:, 1])
        llhs_perm = permutedims(llhs, [k, j, 3])
        buffer = similar(llhs_perm[:, :, 1])

        @views for n in 1:N
            broadcast!(*, buffer_k, m2k[:, n], m3k[:, n])
            broadcast!(*, buffer, llhs_perm[:, :, n], buffer_k)
            sum!(m1j[:, n], buffer')
        end

        m2j[:, 1] = fsm.α̂
        buffer = similar(m1j[:, 1]) # NOT OPTIMAL, m2j should be used instead but we have issue with Julia 1.7 -> 1.8 fixed it, but introduce another bugs
        # check https://github.com/JuliaSparse/SparseArrays.jl/issues/251
        @views for n in 2:N
            broadcast!(*, buffer, m1j[:, n - 1], m2j[:, n - 1])
            mul!(m2j[:, n], T̂ᵀ, buffer)
        end

        @views fill!(m3j[:, N], one(K))
        @views for n in N-1:-1:1
            broadcast!(*, buffer, m1j[:, n + 1], m3j[:, n + 1])
            mul!(m3j[:, n], T̂, buffer)
        end
    end

    return (m1=m1, m2=m2, m3=m3)
end
