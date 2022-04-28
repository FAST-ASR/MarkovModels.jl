# SPDX-License-Identifier: MIT

const Ss = [LogSemiring, ProbSemiring, TropicalSemiring]
const Ts = [Float32, Float64]

function fsmequal(fsm1::FSM, fsm2::FSM)
    n = max(nstates(fsm1), nstates(fsm2)) + 1
    ls1 = totallabelsum(fsm1, n)
    ls2 = totallabelsum(fsm2, n)
    ws1 = totalsum(fsm1, n)
    ws2 = totalsum(fsm2, n)
    ls1 == ls2 && ws1 ≈ ws2
end

@testset "label" begin
    for x in [1, :e, "a"]
        @test Label(x) == UnionConcatSemiring(Set([SymbolSequence([x])]))
    end
end

@testset "constructor" begin
    for S in Ss, T in Ts
        K = S{T}
        α = [1 => one(K)]
        T = [(1, 1) => one(K), (1, 2) => one(K), (2, 2) => one(K),
             (2, 1) => one(K)]
        ω = [2 => one(K)]
        fsm1 = FSM(2, α, T, ω, [Label(1), Label(2)])
        fsm2 = FSM(
            sparsevec([1], [one(K)], 2),
            sparse([1, 1, 2, 2], [1, 2, 2, 1], ones(K, 4), 2, 2),
            sparsevec([2], [one(K)], 2),
            [Label(1), Label(2)]
        )
        @test fsmequal(fsm1, fsm2)
    end
end

@testset "union" begin
    for S in Ss, T in Ts
        K = S{T}
        fsm1 = FSM(
            3,
            [1 => one(K)],
            [(1, 1) => one(K), (1, 2) => one(K), (2, 2) => one(K),
             (2, 3) => one(K), (3, 3) => one(K)],
            [3 => one(K)],
            [Label(1), Label(2), Label(3)]
        )
        fsm2 = FSM(
            3,
            [1 => one(K)],
            [(1, 1) => one(K), (1, 2) => one(K), (2, 2) => one(K),
             (2, 3) => one(K), (3, 3) => one(K)],
            [3 => one(K)],
            [Label(4), Label(5), Label(6)]
        )
        fsm3 = FSM(
            6,
            [1 => one(K), 4 => one(K)],
            [(1, 1) => one(K), (1, 2) => one(K), (2, 2) => one(K),
             (2, 3) => one(K), (3, 3) => one(K),
             (4, 4) => one(K), (4, 5) => one(K), (5, 5) => one(K),
             (5, 6) => one(K), (6, 6) => one(K)],
            [3 => one(K), 6 => one(K)],
            [Label(1), Label(2), Label(3), Label(4), Label(5), Label(6)]
        )
        fsm = union(fsm1, fsm2)
        @test fsmequal(fsm, fsm3)
        @test length(nonzeros(fsm.α)) == length(nonzeros(fsm3.α))
        @test length(nonzeros(fsm.T)) == length(nonzeros(fsm3.T))
        @test length(nonzeros(fsm.ω)) == length(nonzeros(fsm3.ω))
    end
end

@testset "concat" begin
    for S in Ss, T in Ts
        K = S{T}
        fsm1 = FSM(
            3,
            [1 => one(K)],
            [(1, 1) => one(K), (1, 2) => one(K), (2, 2) => one(K),
             (2, 3) => one(K), (3, 3) => one(K)],
            [3 => one(K)],
            [Label(1), Label(2), Label(3)]
        )
        fsm2 = FSM(
            3,
            [1 => one(K)],
            [(1, 1) => one(K), (1, 2) => one(K), (2, 2) => one(K),
             (2, 3) => one(K), (3, 3) => one(K)],
            [3 => one(K)],
            [Label(4), Label(5), Label(6)]
        )
        fsm3 = FSM(
            6,
            [1 => one(K)],
            [(1, 1) => one(K), (1, 2) => one(K), (2, 2) => one(K),
             (2, 3) => one(K), (3, 3) => one(K), (3, 4) => one(K),
             (4, 4) => one(K), (4, 5) => one(K), (5, 5) => one(K),
             (5, 6) => one(K), (6, 6) => one(K)],
            [6 => one(K)],
            [Label(1), Label(2), Label(3), Label(4), Label(5), Label(6)]
        )
        fsm = concat(fsm1, fsm2)
        @test fsmequal(fsm, fsm3)
        @test length(nonzeros(fsm.α)) == length(nonzeros(fsm3.α))
        @test length(nonzeros(fsm.T)) == length(nonzeros(fsm3.T))
        @test length(nonzeros(fsm.ω)) == length(nonzeros(fsm3.ω))
    end
end

@testset "renormalize" begin
    for S in Ss, T in Ts
        K = S{T}
        fsm1 = FSM(
            3,
            [1 => one(K) + one(K)],
            [(1, 1) => one(K), (1, 2) => one(K), (2, 2) => one(K),
             (2, 3) => one(K), (3, 3) => one(K)],
            [3 => one(K)],
            [Label(1), Label(2), Label(3)]
        )
        Z = one(K) + one(K)
        fsm2 = FSM(
            3,
            [1 => one(K)],
            [(1, 1) => one(K) / Z,
             (1, 2) => one(K) / Z,
             (2, 2) => one(K) / Z,
             (2, 3) => one(K) / Z,
             (3, 3) => one(K) / Z],
            [3 => one(K) / Z],
            [Label(1), Label(2), Label(3)]
        )
        fsm = renorm(fsm1)
        @test fsmequal(fsm, fsm2)
        @test length(nonzeros(fsm.α)) == length(nonzeros(fsm2.α))
        @test length(nonzeros(fsm.T)) == length(nonzeros(fsm2.T))
        @test length(nonzeros(fsm.ω)) == length(nonzeros(fsm2.ω))
    end
end

@testset "reversal" begin
    for S in Ss, T in Ts
        K = S{T}
        fsm1 = FSM(
            3,
            [1 => one(K)],
            [(1, 1) => one(K), (1, 2) => one(K), (2, 2) => one(K),
             (2, 3) => one(K), (3, 3) => one(K)],
            [3 => one(K)],
            [Label(1), Label(2), Label(3)]
        )
        Z = one(K) + one(K)
        fsm2 = FSM(
            3,
            [3 => one(K)],
            [(1, 1) => one(K),
             (2, 1) => one(K),
             (2, 2) => one(K),
             (3, 2) => one(K),
             (3, 3) => one(K) ],
            [1 => one(K)],
            [Label(1), Label(2), Label(3)]
        )
        fsm = fsm1'
        @test fsmequal(fsm, fsm2)
        @test fsmequal(fsm', fsm1)
    end
end

@testset "compose" begin
    for S in Ss, T in Ts
        K = S{T}
        fsm1 = FSM(
            3,
            [1 => one(K)],
            [(1, 1) => one(K), (1, 2) => one(K), (2, 2) => one(K),
             (2, 3) => one(K), (3, 3) => one(K)],
            [3 => one(K)],
            [Label(1), Label(2), Label(3)]
        )
        fsm2 = FSM(
            2,
            [1 => one(K)],
            [(1, 1) => one(K), (1, 2) => one(K),
             (2, 2) => one(K), (2, 1) => one(K)],
            [2 => one(K)],
            [Label(:a), Label(:b)]
        )
        fsm3 = FSM(
            6,
            [1 => one(K)],
            [(1, 1) => one(K), (1, 2) => one(K), (2, 2) => one(K),
             (2, 3) => one(K), (3, 3) => one(K),
             (4, 4) => one(K), (4, 5) => one(K), (5, 5) => one(K),
             (5, 6) => one(K), (6, 6) => one(K),
             (3, 1) => one(K), (3, 4) => one(K),
             (6, 1) => one(K), (6, 4) => one(K)],
            [6 => one(K)],
            [Label(:a) * Label(1), Label(:a) * Label(2), Label(:a) * Label(3),
             Label(:b) * Label(1), Label(:b) * Label(2), Label(:b) * Label(3)]
        )
        fsm = fsm2 ∘ [fsm1, fsm1]
        @test fsmequal(fsm, fsm3)
        @test length(nonzeros(fsm.α)) == length(nonzeros(fsm3.α))
        @test length(nonzeros(fsm.T)) == length(nonzeros(fsm3.T))
        @test length(nonzeros(fsm.ω)) == length(nonzeros(fsm3.ω))
    end
end
#
#
#@testset "determinize" begin
#    SR = ProbabilitySemifield{Float64}
#
#    fsm = VectorFSM{SR}()
#    s1 = addstate!(fsm, "a"; initweight = SR(0.5))
#    s2 = addstate!(fsm, "b"; initweight = SR(0.5))
#    s3 = addstate!(fsm, "c"; finalweight = SR(0.25))
#    addarc!(fsm, s1, s1, SR(0.75))
#    addarc!(fsm, s1, s2, SR(0.25))
#    addarc!(fsm, s2, s2, SR(0.75))
#    addarc!(fsm, s2, s3, SR(0.25))
#    addarc!(fsm, s3, s3, SR(0.75))
#    @test fsmequal(fsm, fsm |> determinize)
#
#    fsm = VectorFSM{SR}()
#    s1 = addstate!(fsm, "a"; initweight = one(SR), finalweight = one(SR))
#    s2 = addstate!(fsm, "b")
#    s3 = addstate!(fsm, "b")
#    s4 = addstate!(fsm, "c", finalweight = one(SR))
#    addarc!(fsm, s1, s2)
#    addarc!(fsm, s1, s3)
#    addarc!(fsm, s2, s1)
#    addarc!(fsm, s1, s4)
#    @test fsmequal(fsm |> renormalize, fsm |> determinize)
#end
#
#@testset "minimize" begin
#    SR = ProbabilitySemifield{Float64}
#
#    fsm = VectorFSM{SR}()
#    s1 = addstate!(fsm, "a"; initweight = SR(0.5))
#    s2 = addstate!(fsm, "b"; initweight = SR(0.5))
#    s3 = addstate!(fsm, "c"; finalweight = SR(0.25))
#    s4 = addstate!(fsm, "a"; initweight = SR(0.5))
#    s5 = addstate!(fsm, "d")
#    s6 = addstate!(fsm, "c"; finalweight = SR(0.25))
#    s7 = addstate!(fsm, "e")
#    addarc!(fsm, s1, s1, SR(0.75))
#    addarc!(fsm, s1, s2, SR(0.25))
#    addarc!(fsm, s2, s2, SR(0.75))
#    addarc!(fsm, s2, s3, SR(0.25))
#    addarc!(fsm, s3, s3, SR(0.75))
#    addarc!(fsm, s4, s4, SR(0.75))
#    addarc!(fsm, s4, s5, SR(0.25))
#    addarc!(fsm, s5, s5, SR(0.75))
#    addarc!(fsm, s5, s6, SR(0.25))
#    addarc!(fsm, s6, s6, SR(0.75))
#    addarc!(fsm, s7, s7, SR(0.75))
#    addarc!(fsm, s4, s7, SR(0.75))
#    addarc!(fsm, s7, s3, SR(0.25))
#
#    mfsm = VectorFSM{SR}()
#    s1 = addstate!(mfsm, "a"; initweight = SR(1.0))
#    s2 = addstate!(mfsm, "b"; initweight = SR(0.5))
#    s3 = addstate!(mfsm, "e")
#    s4 = addstate!(mfsm, "d")
#    s5 = addstate!(mfsm, "c"; finalweight = SR(0.5))
#    addarc!(mfsm, s1, s1, SR(1.5))
#    addarc!(mfsm, s1, s2, SR(0.25))
#    addarc!(mfsm, s1, s3, SR(0.75))
#    addarc!(mfsm, s1, s4, SR(0.25))
#    addarc!(mfsm, s2, s2, SR(0.75))
#    addarc!(mfsm, s2, s5, SR(0.25))
#    addarc!(mfsm, s3, s3, SR(0.75))
#    addarc!(mfsm, s3, s5, SR(0.25))
#    addarc!(mfsm, s4, s4, SR(0.75))
#    addarc!(mfsm, s4, s5, SR(0.25))
#    addarc!(mfsm, s5, s5, SR(1.5))
#    mfsm = mfsm |> renormalize
#
#    @test fsmequal(mfsm, fsm |> minimize)
#end
#
#@testset "transpose" begin
#    SR = ProbabilitySemifield{Float64}
#
#    fsm = VectorFSM{SR}()
#    s1 = addstate!(fsm, "a"; initweight = SR(0.5))
#    s2 = addstate!(fsm, "b"; initweight = SR(0.5))
#    s3 = addstate!(fsm, "c"; finalweight = SR(0.25))
#    addarc!(fsm, s1, s1, SR(0.75))
#    addarc!(fsm, s1, s2, SR(0.25))
#    addarc!(fsm, s2, s2, SR(0.75))
#    addarc!(fsm, s2, s3, SR(0.25))
#    addarc!(fsm, s3, s3, SR(0.75))
#
#    rfsm = VectorFSM{SR}()
#    s1 = addstate!(rfsm, "a"; finalweight = SR(0.5))
#    s2 = addstate!(rfsm, "b"; finalweight = SR(0.5))
#    s3 = addstate!(rfsm, "c"; initweight = SR(0.25))
#    addarc!(rfsm, s1, s1, SR(0.75))
#    addarc!(rfsm, s2, s1, SR(0.25))
#    addarc!(rfsm, s2, s2, SR(0.75))
#    addarc!(rfsm, s3, s2, SR(0.25))
#    addarc!(rfsm, s3, s3, SR(0.75))
#    @test fsmequal(rfsm, fsm |> transpose)
#    @test fsmequal(fsm, fsm |> transpose |> transpose)
#end
#
#@testset "HierarchicalFSM" begin
#    SR = ProbabilitySemifield{Float64}
#
#    fsma = VectorFSM{SR}()
#    s1 = addstate!(fsma, "a"; initweight = SR(0.5))
#    s2 = addstate!(fsma, "b"; initweight = SR(0.5))
#    s3 = addstate!(fsma, "c"; finalweight = SR(0.25))
#    addarc!(fsma, s1, s1, SR(0.75))
#    addarc!(fsma, s1, s2, SR(0.25))
#    addarc!(fsma, s2, s2, SR(0.75))
#    addarc!(fsma, s2, s3, SR(0.25))
#    addarc!(fsma, s3, s3, SR(0.75))
#
#    fsmb = VectorFSM{SR}()
#    s1 = addstate!(fsmb, "d"; initweight = SR(0.5))
#    s2 = addstate!(fsmb, "e"; initweight = SR(0.5))
#    s3 = addstate!(fsmb, "f"; finalweight = SR(0.25))
#    addarc!(fsmb, s1, s1, SR(0.75))
#    addarc!(fsmb, s1, s2, SR(0.25))
#    addarc!(fsmb, s2, s2, SR(0.75))
#    addarc!(fsmb, s2, s3, SR(0.25))
#    addarc!(fsmb, s3, s3, SR(0.75))
#
#    fsm = VectorFSM{SR}()
#    s1 = addstate!(fsm, "x"; initweight = one(SR))
#    s2 = addstate!(fsm, "y"; initweight = one(SR))
#    s3 = addstate!(fsm, "z"; finalweight = one(SR))
#    addarc!(fsm, s1, s2)
#    addarc!(fsm, s2, s3)
#    addarc!(fsm, s3, s1)
#    addarc!(fsm, s1, s3)
#    fsm = fsm |> renormalize
#
#    hfsm = VectorFSM{SR}()
#    s1 = addstate!(hfsm, ("x", "a"); initweight = SR(0.25))
#    s2 = addstate!(hfsm, ("x", "b"); initweight = SR(0.25))
#    s3 = addstate!(hfsm, ("x", "c"))
#    s4 = addstate!(hfsm, ("y", "d"); initweight = SR(0.25))
#    s5 = addstate!(hfsm, ("y", "e"); initweight = SR(0.25))
#    s6 = addstate!(hfsm, ("y", "f"))
#    s7 = addstate!(hfsm, ("z", "a"))
#    s8 = addstate!(hfsm, ("z", "b"))
#    s9 = addstate!(hfsm, ("z", "c"); finalweight = SR(0.125))
#    addarc!(hfsm, s1, s1, SR(0.75))
#    addarc!(hfsm, s1, s2, SR(0.25))
#    addarc!(hfsm, s2, s2, SR(0.75))
#    addarc!(hfsm, s2, s3, SR(0.25))
#    addarc!(hfsm, s3, s3, SR(0.75))
#    addarc!(hfsm, s3, s4, SR(0.0625))
#    addarc!(hfsm, s3, s5, SR(0.0625))
#    addarc!(hfsm, s3, s7, SR(0.0625))
#    addarc!(hfsm, s3, s8, SR(0.0625))
#    addarc!(hfsm, s4, s4, SR(0.75))
#    addarc!(hfsm, s4, s5, SR(0.25))
#    addarc!(hfsm, s5, s5, SR(0.75))
#    addarc!(hfsm, s5, s6, SR(0.25))
#    addarc!(hfsm, s6, s6, SR(0.75))
#    addarc!(hfsm, s6, s7, SR(0.125))
#    addarc!(hfsm, s6, s8, SR(0.125))
#    addarc!(hfsm, s7, s7, SR(0.75))
#    addarc!(hfsm, s7, s8, SR(0.25))
#    addarc!(hfsm, s8, s8, SR(0.75))
#    addarc!(hfsm, s8, s9, SR(0.25))
#    addarc!(hfsm, s9, s9, SR(0.75))
#    addarc!(hfsm, s9, s1, SR(0.0625))
#    addarc!(hfsm, s9, s2, SR(0.0625))
#
#    smap = Dict("x" => fsma, "y" => fsmb, "z" => fsma)
#    @test fsmequal(hfsm, HierarchicalFSM(fsm, smap))
#    @test length(HierarchicalFSM(fsm, smap)) == length(hfsm)
#end
#
#@testset "MatrixFSM" begin
#    SR = ProbabilitySemifield{Float64}
#
#    fsm = VectorFSM{SR}()
#    s1 = addstate!(fsm, ("a", 1); initweight = SR(0.5))
#    s2 = addstate!(fsm, ("b", 1); initweight = SR(0.5))
#    s3 = addstate!(fsm, ("c", 1); finalweight = SR(0.25))
#    addarc!(fsm, s1, s1, SR(0.75))
#    addarc!(fsm, s1, s2, SR(0.25))
#    addarc!(fsm, s2, s2, SR(0.75))
#    addarc!(fsm, s2, s3, SR(0.25))
#    addarc!(fsm, s3, s3, SR(0.75))
#
#    pdfid_mapping = Dict(
#        "a" => 1,
#        "b" => 2,
#        "c" => 3
#    )
#    mfsm = MatrixFSM(fsm, pdfid_mapping, l -> l[1])
#
#    @test fsmequal(fsm, mfsm)
#    @test length(mfsm) == length(fsm)
#
#    ufsm = VectorFSM{SR}()
#    for i in 1:3
#        s1 = addstate!(ufsm, ("a", 1); initweight = SR(0.5))
#        s2 = addstate!(ufsm, ("b", 1); initweight = SR(0.5))
#        s3 = addstate!(ufsm, ("c", 1); finalweight = SR(0.25))
#        addarc!(ufsm, s1, s1, SR(0.75))
#        addarc!(ufsm, s1, s2, SR(0.25))
#        addarc!(ufsm, s2, s2, SR(0.75))
#        addarc!(ufsm, s2, s3, SR(0.25))
#        addarc!(ufsm, s3, s3, SR(0.75))
#    end
#    @test fsmequal(ufsm, union(mfsm, mfsm, mfsm))
#
#    ufsm = union(mfsm, mfsm, mfsm)
#    if CUDA.functional()
#        @test length(gpu(mfsm).π) == length(mfsm.π)
#        @test length(gpu(ufsm).π) == length(ufsm.π)
#    end
#end
