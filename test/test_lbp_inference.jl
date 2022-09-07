const N = 100 # number of frames
const S1 = 30 # number of states for spkr1
const S2 = 33 # number of states for spkr2
const T = Float32
const SF = LogSemiring{T} 

function naive_lbp_step(messages, ffsm::FactorialFSM{T}, llhs::AbstractArray{T, 3}) where T
    N = size(llhs, 3)
    n_spkrs = 2
    m1, m2, m3 = deepcopy(messages)

    for j in 1:n_spkrs

        for n in 1:N
            buffer = llhs[:, :, n]

            for k in 1:n_spkrs
                k == j && continue
                if k == 2
                    buffer = permutedims(buffer, [2, 1])
                    buffer = buffer .* (m2[k][:, n] .* m3[k][:, n])
                    buffer = permutedims(buffer, [2, 1])
                elseif k == 1
                    buffer = buffer .* (m2[k][:, n] .* m3[k][:, n])
                else
                    throw(ErrorException("Not available for more then 2 spkrs"))
                end
            end
            m1[j][:, n] = sum(buffer, dims=[k for k in 1:n_spkrs if k != j])
        end

        fsm = ffsm.fsms[j]
        m2[j][:, 1] = fsm.α̂
        for n in 2:N
            m2[j][:, n] = ((m2[j][:, n - 1] .* m1[j][:, n - 1])' * fsm.T̂)'
        end

        fill!(m3[j][:, 1], one(T))
        for n in N-1:-1:1
            m3[j][:, n] = fsm.T̂ * (m3[j][:, n + 1] .* m1[j][:, n + 1])
        end
    end

    return (m1=m1, m2=m2, m3=m3)
end

make_lin_ffsm(SF, T, num_states_per_fsm...) = begin
    fsms = FSM{SF}[]
    smaps = AbstractMatrix{SF}[]
    for S in num_states_per_fsm
        α = sparse(vcat(one(T), zeros(T, S-2)))
        T̂ = sparse(Bidiagonal([T(0.75) for _ in 1:S-1], [T(0.25) for _ in 1:S-2], :U))
        ω = sparse(vcat(zeros(T, S-2), [T(0.25)]))
        labels = collect(1:S-1)
        push!(
            fsms,
            FSM(
                convert.(SF, log.(α)),
                convert.(SF, log.(T̂)),
                convert.(SF, log.(ω)),
                labels
            ) |> renorm
        )
        push!(smaps, ones(S,S))
    end
    FactorialFSM(fsms, smaps)
end

make_ffsm(SF, T, num_states_per_fsm...) = begin
    fsms = FSM{SF}[]
    smaps = AbstractMatrix{SF}[]
    for S in num_states_per_fsm
        α = sprand(T, S-1, 0.25)
        T̂ = sprand(T, S-1, S-1, 0.95)
        ω = sprand(T, S-1, 0.75)
        labels = collect(1:S-1)
        push!(
            fsms,
            FSM(
                convert.(SF, log.(α)),
                convert.(SF, log.(T̂)),
                convert.(SF, log.(ω)),
                labels
            ) |> renorm
        )
        push!(smaps, ones(S,S))
    end
    FactorialFSM(fsms, smaps)
end

false_print(msg) = begin 
    println(msg)
    println("")
    false
end

@testset "random FactorialFSM" begin
    ffsm = make_ffsm(SF, T, S1, S2)
    llhs = convert.(SF, log.(rand(T, S1, S2, N)))

    m1 = [Array{SF}(undef, S, N) for S in [S1, S2]]
    m2 = [ones(SF, S, N-1) for S in [S1, S2]]
    m2 = [hcat(fsm.α̂, m) for (fsm, m) in zip(ffsm.fsms, m2)]
    m3 = [ones(SF, S, N) for S in [S1, S2]]

    ref_m1, ref_m2, ref_m3 = naive_lbp_step([m1,m2,m3], ffsm, llhs)
    hyp_m1, hyp_m2, hyp_m3 = lbp_step!(deepcopy((m1=m1, m2=m2, m3=m3)), ffsm, llhs)

    for j in 1:2
        @test all(isapprox.(val.(ref_m1[j]), val.(hyp_m1[j]), nans=true))
        @test all(isapprox.(val.(ref_m2[j]), val.(hyp_m2[j]), nans=true))
        @test all(isapprox.(val.(ref_m3[j]), val.(hyp_m3[j]), nans=true))
        #@test all(ref_m2[j] .≈ hyp_m2[j]) || false_print((println.(ref_m2[j] .≈ hyp_m2[j]), println.(ref_m2[j]), println(""), println.( hyp_m2[j])))
    end
end

@testset "Linear Factorial FSM" begin
    # Linear FSM
    lffsm = make_lin_ffsm(SF, T, S1, S2)
    llhs = convert.(SF, log.(rand(T, S1, S2, N)))

    m1 = [Array{SF}(undef, S, N) for S in [S1, S2]]
    m2 = [ones(SF, S, N-1) / SF(S) for S in [S1, S2]]
    m2 = [hcat(fsm.α̂, m) for (fsm, m) in zip(lffsm.fsms, m2)]
    m3 = [ones(SF, S, N) / SF(S) for S in [S1, S2]]

    ref_m1, ref_m2, ref_m3 = naive_lbp_step([m1,m2,m3], lffsm, llhs)
    hyp_m1, hyp_m2, hyp_m3 = lbp_step!(deepcopy((m1=m1, m2=m2, m3=m3)), lffsm, llhs)

    for j in 1:2
        @test all(isapprox.(val.(ref_m1[j]), val.(hyp_m1[j]), nans=true))
        @test all(isapprox.(val.(ref_m2[j]), val.(hyp_m2[j]), nans=true))
        @test all(isapprox.(val.(ref_m3[j]), val.(hyp_m3[j]), nans=true))
        #@test all(ref_m2[j] .≈ hyp_m2[j]) || false_print((println.(ref_m2[j] .≈ hyp_m2[j]), println.(ref_m2[j]), println(""), println.( hyp_m2[j])))
    end
end
