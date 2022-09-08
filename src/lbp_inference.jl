struct FactorialFSM{K<:Semiring}
    fsms::Vector{FSM{K}}
    smap::Vector{AbstractSparseMatrix{K}}
end

function FactorialFSM(
            fsm1::FSM{K}, smap1::AbstractSparseMatrix{K},
            fsm2::FSM{K}, smap2::AbstractSparseMatrix{K}
) where K
    FactorialFSM([fsm1, fsm2], [smap1, smap2])
end

nfsms(ffsm::FactorialFSM{K}) where K = length(ffsm.fsms)
getindex(ffsm::FactorialFSM{K}, key::Integer) where K = ffsm.fsms[key]

function joint_smap(smap1::AbstractSparseMatrix{K}, smap2::AbstractSparseMatrix{K}) where K
    S1, P1 = size(smap1)
    S2, P2 = size(smap2)
    @assert P1 == P2

    I, J, V = [], [], []
    for (i1, j1, v2) in zip(findnz(smap1))
        for (i2, j2, v2) in zip(findnz(smap2))
            push!(I, (i2-1) * S1 + i1)
            push!(J, (j2-1) * P1 + j1)
            push!(V, one(K))
        end
    end
    sparse(I, J, V, S1*S2, P1*P2)
end

function init_messages(ffsm::FactorialFSM{K}, N::Integer) where K
    S = [nstates(fsm) + 1 for fsm in ffsm.fsms] # + virtual state

    m1 = [Array{K}(undef, s, N) for s in S]
    m2 = [Array{K}(undef, s, N) for s in S]
    m3 = [Array{K}(undef, s, N) for s in S]

    @views for j in 1:length(S)
        fsm = ffsm[j]
        S = nstates(fsm)
        m1j, m2j, m3j = m1[j], m2[j], m3[j] 
        fill!(m2j[:, 2:end], one(K) / K(S))
        m2j[:, 1] = fsm.α̂
        fill!(m3j, one(K) / K(S))
    end

    (m1 = m1, m2 = m2, m3 = m3)
end

function compare_msgs(old_mgs, new_msg; eps=1e-4)
    n_spkrs = length(old_mgs[:m1])
    changes = []
    for n_msg in 1:3
        om, nm = old_mgs[Symbol("m", n_msg)], new_msg[Symbol("m", n_msg)]

        for j in 1:n_spkrs
            diff = isapprox.(val.(om[j]), val.(nm[j]); atol=eps)
            diff_N = findall(vec(.!all(diff, dims=1)))
            n_diff_N = length(diff_N)
            append!(changes, zip(repeat([j], n_diff_N), repeat([n_msg], n_diff_N), diff_N))
        end
    end
    changes
end

"""
    pdfposteriors(ffsm::FactorialFSM{K}, llhs::AbstractMatrix{K}; eps=1e-4, max_iter=10) where {K<:Semiring}

Compute pdf posteriors given loglikehood `llhs` and FactorialFSM `ffsm`.
Likehoods `llhs` are represented as 2D matrix of shape
`(S1*S2, N)`, where `S1` is number of states in 1st FSM and `S2` is number
of states in 2nd FSM in `ffsm`.

args:
    ffsm - FactorialFSM od 2 FSMs with states `S1` and `S2`
    llhs - loglikehoods with size `(S1*S2, N)`
"""
function pdfposteriors(ffsm::FactorialFSM{K}, llhs::AbstractMatrix{K}; eps=1e-4, max_iter=10) where {K<:Semiring}
    n_spkrs = nfsms(ffsm)
    states_per_fsm = [nstates(ffsm[i]) + 1 for i in 1:n_spkrs] # + virtual state
    N = size(llhs, 2)
    total_states = reduce(*, states_per_fsm)
    # we assume that llhs is already expanded (see `expand`)
    state_llhs = joint_smap(ffsm.smap...) * llhs
    @assert size(state_llhs, 1) == total_states
    state_llhs = reshape(state_llhs, vcat(states_per_fsm, [N])...) # S1 x S2 x ... x N

    messages = init_messages(ffsm, N)
    iter = 0
    changes = nothing

    while true
        iter += 1
        new_messages = lbp_step!(deepcopy(messages), ffsm, state_llhs)
        # check the difference between messages
        changes = compare_msgs(messages, new_messages; eps=eps)
        messages = new_messages
        # if all messages are same then break
        if isempty(changes) || iter >= max_iter
            break
        end
    end
    total_num_msgs = [size.(m, 2) for m in messages] |> sum |> sum
    println("Finished in iter: $iter (max: $max_iter)")
    println("Number of changed msgs before max_iter was reached: $(length(changes)) ($total_num_msgs)")

    m1, m2, m3 = messages
    result = []
    ttl = zero(K)
    for (j, (m1j, m2j, m3j)) in enumerate(zip(m1, m2, m3))
        state_marginals = broadcast!(*, m1j, m1j, m2j, m3j)
        pdf_marginals = ffsm.smap[j]' * state_marginals
        sums = sum(pdf_marginals, dims=1) # 1 x N
        broadcast!(/, pdf_marginals, pdf_marginals, sums)
        ttl += minimum(sums)
        push!(result, pdf_marginals)
    end

    # TODO: return pdf marginals of the same size as llhs 
    result, ttl
end

function lbp_step!(messages, ffsm::FactorialFSM{K}, llhs::AbstractArray{K, 3}) where K
    n_spkrs = ndims(llhs) - 1
    @assert n_spkrs == 2 "Currently we do not support more than 2 speakers!"

    m1, m2, m3 = messages
    N = size(llhs, ndims(llhs))

    for j in 1:n_spkrs
        # this spkr's messages
        m1j, m2j, m3j = m1[j], m2[j], m3[j]
        fsm = ffsm[j]
        T̂, T̂ᵀ = fsm.T̂, permutedims(fsm.T̂, [2,1]) # TODO maybe not optimal

        # other spkr's messages
        k = n_spkrs - j + 1
        m2k, m3k = m2[k], m3[k]
        buffer_k = similar(m2k[:, 1])
        llhs_perm = permutedims(llhs, [j, k, 3])

        @views for n in 1:N
            broadcast!(*, buffer_k, m2k[:, n], m3k[:, n])
            broadcast!(/, buffer_k, buffer_k, sum(buffer_k))
            mul!(m1j[:, n], llhs_perm[:, :, n], buffer_k)
            #broadcast!(*, buffer, llhs_perm[:, :, n], buffer_k)
            #sum!(m1j[:, n], buffer')
        end

        m2j[:, 1] = fsm.α̂
        buffer = similar(m1j[:, 1]) # NOT OPTIMAL, m2j should be used instead
        # but we have issue with Julia 1.7 -> 1.8 fixed it, but introduce another bugs
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
