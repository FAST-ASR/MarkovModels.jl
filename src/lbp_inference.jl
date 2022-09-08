struct CompiledFSM{K<:Semiring}
    α̂
    T̂
    T̂ᵀ
    Ĉ
    Ĉᵀ
end
compile(fsm::FSM, Ĉ::AbstractMatrix) = CompiledFSM{eltype(fsm.α̂)}(fsm.α̂, fsm.T̂, copy(fsm.T̂'), Ĉ, copy(Ĉ'))
nstates(fsm::CompiledFSM) = length(fsm.α̂) - 1

function Adapt.adapt_structure(::Type{<:CuArray}, cfsm::CompiledFSM{K}) where K
    T̂ = CuSparseMatrixCSC(cfsm.T̂)
    T̂ᵀ = CuSparseMatrixCSC(cfsm.T̂ᵀ)
    Ĉ = CuSparseMatrixCSC(cfsm.Ĉ)
    Ĉᵀ = CuSparseMatrixCSC(cfsm.Ĉᵀ)
    CompiledFSM{K}(
        CuSparseVector(cfsm.α̂),
        CuSparseMatrixCSR(T̂ᵀ.colPtr, T̂ᵀ.rowVal, T̂ᵀ.nzVal, T̂.dims),
        CuSparseMatrixCSR(T̂.colPtr, T̂.rowVal, T̂.nzVal, T̂ᵀ.dims),
        CuSparseMatrixCSR(Ĉᵀ.colPtr, Ĉᵀ.rowVal, Ĉᵀ.nzVal, Ĉ.dims),
        CuSparseMatrixCSR(Ĉ.colPtr, Ĉ.rowVal, Ĉ.nzVal, Ĉᵀ.dims),
    )
end

struct FactorialFSM{K<:Semiring}
    fsms::Vector{FSM{K}}
end

struct CompiledFactorialFSM{K<:Semiring, J}
    fsms::Vector{CompiledFSM{K}}
    Ĉ::AbstractSparseMatrix{K}
    Ĉᵀ::AbstractSparseMatrix{K}
end

function compile(ffsm::FactorialFSM{K}, Ĉs::AbstractMatrix{K}...) where K
    J = length(ffsm.fsms)
    @assert length(Ĉs) == J "Number of state maps `Ĉs` has to be same as number of FSMs in FactorialFSM"
    Ĉ = foldl(joint_smap, sparse.(Ĉs))
    CompiledFactorialFSM{K, J}(
        map(zip(ffsm.fsms, Ĉs)) do (fsm, smap) compile(fsm, smap) end,
        #[compile(fsm, smap) for (fsm, smap) in zip(ffsm.fsms, Ĉs)],
        Ĉ,
        copy(Ĉ')
    )
end

Base.getindex(cffsm::CompiledFactorialFSM{K}, key::Integer) where K = cffsm.fsms[key]
nfsms(cffsm::CompiledFactorialFSM{K, J}) where {K, J} = J

function Adapt.adapt_structure(T::Type{<:CuArray}, cffsm::CompiledFactorialFSM{K}) where K
    fsms = adapt_structure.(T, cffsm.fsms)
    Ĉ = CuSparseMatrixCSC(cffsm.Ĉ)
    Ĉᵀ = CuSparseMatrixCSC(cffsm.Ĉᵀ)
    CompiledFactorialFSM(
        fsms,
        CuSparseMatrixCSR(Ĉᵀ.colPtr, Ĉᵀ.rowVal, Ĉᵀ.nzVal, Ĉ.dims),
        CuSparseMatrixCSR(Ĉ.colPtr, Ĉ.rowVal, Ĉ.nzVal, Ĉᵀ.dims),
    )
end

function batch(cffsm1::CompiledFactorialFSM{K, J}, cffsms::CompiledFactorialFSM{K, J}...) where {K, J}
    fsms = Vector{CompiledFSM{K}}[]
    for j in 1:J
        push!(fsms, batch(cffsm1.fsms[j], map(cffsm -> cffsm.fsms[j], cffsms)...))
    end
    CompiledFactorialFSM{K, J}(
        fsms,
        blockdiag(cffsm1.Ĉ, map(cffsm -> cffsm.Ĉ, cffsms)...),
        blockdiag(cffsm1.Ĉᵀ, map(cffsm -> cffsm.Ĉᵀ, cffsms)...)
    )
end

function joint_smap(smap1::AbstractSparseMatrix{K}, smap2::AbstractSparseMatrix{K}) where K
    S1, P1 = size(smap1)
    S2, P2 = size(smap2)
    @assert P1 == P2

    I, J, V = Int[], Int[], K[]
    for (i1, j1, v2) in zip(findnz(smap1)...)
        for (i2, j2, v2) in zip(findnz(smap2)...)
            push!(I, (i2-1) * S1 + i1)
            push!(J, (j2-1) * P1 + j1)
            push!(V, one(K))
        end
    end
    sparse(I, J, V, S1*S2, P1*P2)
end

function init_messages(ffsm::CompiledFactorialFSM{K, J}, N::Integer) where {K,J}
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

function compare_msgs(old_mgs, new_msg; eps=1.2)
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
    V̂s - B-element vector of expanded loglikehoods with size `(S1*S2, N)` (see `expand`)
"""
function pdfposteriors(ffsm::CompiledFactorialFSM{K, J}, V̂s::Vector{<:AbstractMatrix{K}}; eps=1.2, min_iter=3, max_iter=10) where {K<:Semiring, J}
    B = length(V̂s)
    llhs = vcat(V̂s...)
    N = size(llhs, 2)
    n_spkrs = nfsms(ffsm)
    states_per_fsm = [nstates(ffsm[i]) + 1 for i in 1:n_spkrs] # + virtual state
    total_states = reduce(*, states_per_fsm)
    # we assume that llhs is already expanded (see `expand`)
    state_llhs = ffsm.Ĉ * llhs
    @assert size(state_llhs, 1) == total_states "size(state_llhs): $(size(state_llhs, 1)) != $total_states"
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
        if (isempty(changes) && iter >= min_iter) || iter >= max_iter
            break
        end
    end
    total_num_msgs = [size.(m, 2) for m in messages] |> sum |> sum
    println("Finished in iter: $iter (max: $max_iter)")
    println("Number of changed msgs before max_iter was reached: $(length(changes)) ($total_num_msgs)")

    m1, m2, m3 = messages
    result = []
    ttl = zeros(K, B)
    for (j, (m1j, m2j, m3j)) in enumerate(zip(m1, m2, m3))
        state_marginals = broadcast!(*, m1j, m1j, m2j, m3j)
        pdf_marginals = ffsm[j].Ĉᵀ * state_marginals # B*P x N
        pdf_marginals = permutedims(reshape(pdf_marginals, :, B, N), (2, 1, 3)) # B x P x N
        sums = sum(pdf_marginals, dims=2) # B x 1 x N
        broadcast!(/, pdf_marginals, pdf_marginals, sums)
        broadcast!(+, ttl, ttl, dropdims(minimum(sums, dims=(2,3)), dims=(2,3)))
        push!(result, (exp ∘ val).(pdf_marginals[:, 1:end-1, 1:end-1]))
    end

    # TODO: return pdf marginals of the same size as llhs 
    result, val.(ttl)
end

function lbp_step!(messages, ffsm::CompiledFactorialFSM{K, J}, llhs::AbstractArray{K, D}) where {K, J, D}
    n_spkrs = ndims(llhs) - 1
    @assert n_spkrs == 2 "Currently we do not support more than 2 speakers!"

    m1, m2, m3 = messages
    N = size(llhs, ndims(llhs))

    for j in 1:n_spkrs
        # this spkr's messages
        m1j, m2j, m3j = m1[j], m2[j], m3[j]
        fsm = ffsm[j]
        T̂, T̂ᵀ = fsm.T̂, fsm.T̂ᵀ

        # other spkr's messages
        k = n_spkrs - j + 1
        m2k, m3k = m2[k], m3[k]
        buffer_k = similar(m2k[:, 1])
        llhs_perm = permutedims(llhs, [j, k, 3])

        @views for n in 1:N
            broadcast!(*, buffer_k, m2k[:, n], m3k[:, n])
            broadcast!(/, buffer_k, buffer_k, sum(buffer_k)) # Do NOT normalize
            # (w/o norm it will converge faster)
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
