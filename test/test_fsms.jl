# SPDX-License-Identifier: MIT

const weight_semirings = [BoolSemiring, LogSemiring, ProbSemiring,
                          TropicalSemiring]
const parametric_semirings = Set([LogSemiring, ProbSemiring, TropicalSemiring])
const divisible_semirings = Set([LogSemiring, ProbSemiring, TropicalSemiring])
const types = [Float32, Float64]

function fsmequal(fsm1::FSM, fsm2::FSM)
    n = max(nstates(fsm1), nstates(fsm2))
    ls1 = totallabelsum(fsm1, n)
    ls2 = totallabelsum(fsm2, n)
    ws1 = totalweightsum(fsm1, n)
    ws2 = totalweightsum(fsm2, n)
    ls1 == ls2 && isapprox(val(ws1), val(ws2), atol=1e-12)
end

@testset "label" begin
    for x in [1, :e, "a"]
        @test Label(x) == SequenceMonoid(tuple(x))
    end
    @test Label() == one(SequenceMonoid)
end

@testset "constructor" begin
    for S in weight_semirings, T in types
        K = S ∈ parametric_semirings ? S{T} : S
        α = [1 => one(K)]
        T = [(1, 1) => one(K), (1, 2) => one(K), (2, 2) => one(K),
             (2, 1) => one(K)]
        ω = [2 => one(K)]
        fsm1 = FSM(α, T, ω, [Label(1), Label(2)])
        fsm2 = FSM(
            sparsevec([1], [one(K)], 2),
            sparse([1, 1, 2, 2], [1, 2, 2, 1], ones(K, 4), 2, 2),
            sparsevec([2], [one(K)], 2),
            [Label(1), Label(2)]
        )
        @test fsmequal(fsm1, fsm2)

        onestr = "$(val(one(K)))"
        fsm3 = FSM("""
        {
            "semiring": "$K",
            "initstates": [[1, $onestr]],
            "arcs": [[1, 1, $onestr], [1, 2, $onestr], [2, 2, $onestr],
                     [2, 1, $onestr]],
            "finalstates": [[2, $onestr]],
            "labels": [1, 2]
        }
        """)
        @test fsmequal(fsm1, fsm3)

        α = [1 => one(K)]
        T = []
        ω = [1 => one(K)]
        fsm1 = FSM(α, T, ω, [Label(1)])
        fsm2 = FSM(
            sparsevec([1], [one(K)], 1),
            spzeros(K, 1, 1),
            sparsevec([1], [one(K)], 1),
            [Label(1)]
        )
        @test fsmequal(fsm1, fsm2)
    end
end

@testset "union" begin
    for S in weight_semirings, T in types
        K = S ∈ parametric_semirings ? S{T} : S
        fsm1 = FSM(
            [1 => one(K)],
            [(1, 1) => one(K), (1, 2) => one(K), (2, 2) => one(K),
             (2, 3) => one(K), (3, 3) => one(K)],
            [3 => one(K)],
            [Label(1), Label(2), Label(3)]
        )
        fsm2 = FSM(
            [1 => one(K)],
            [(1, 1) => one(K), (1, 2) => one(K), (2, 2) => one(K),
             (2, 3) => one(K), (3, 3) => one(K)],
            [3 => one(K)],
            [Label(4), Label(5), Label(6)]
        )
        fsm3 = FSM(
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

@testset "concatenation" begin
    for S in weight_semirings, T in types
        K = S ∈ parametric_semirings ? S{T} : S
        fsm1 = FSM(
            [1 => one(K)],
            [(1, 1) => one(K), (1, 2) => one(K), (2, 2) => one(K),
             (2, 3) => one(K), (3, 3) => one(K)],
            [3 => one(K)],
            [Label(1), Label(2), Label(3)]
        )
        fsm2 = FSM(
            [1 => one(K)],
            [(1, 1) => one(K), (1, 2) => one(K), (2, 2) => one(K),
             (2, 3) => one(K), (3, 3) => one(K)],
            [3 => one(K)],
            [Label(4), Label(5), Label(6)]
        )
        fsm3 = FSM(
            [1 => one(K)],
            [(1, 1) => one(K), (1, 2) => one(K), (2, 2) => one(K),
             (2, 3) => one(K), (3, 3) => one(K), (3, 4) => one(K),
             (4, 4) => one(K), (4, 5) => one(K), (5, 5) => one(K),
             (5, 6) => one(K), (6, 6) => one(K)],
            [6 => one(K)],
            [Label(1), Label(2), Label(3), Label(4), Label(5), Label(6)]
        )
        fsm = cat(fsm1, fsm2)
        @test fsmequal(fsm, fsm3)
        @test length(nonzeros(fsm.α)) == length(nonzeros(fsm3.α))
        @test length(nonzeros(fsm.T)) == length(nonzeros(fsm3.T))
        @test length(nonzeros(fsm.ω)) == length(nonzeros(fsm3.ω))
    end
end

@testset "renormalize" begin
    for S in divisible_semirings, T in types
        K = S ∈ parametric_semirings ? S{T} : S
        fsm1 = FSM(
            [1 => one(K) + one(K)],
            [(1, 1) => one(K), (1, 2) => one(K), (2, 2) => one(K),
             (2, 3) => one(K), (3, 3) => one(K)],
            [3 => one(K)],
            [Label(1), Label(2), Label(3)]
        )
        Z = one(K) + one(K)
        fsm2 = FSM(
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
    for S in weight_semirings, T in types
        K = S ∈ parametric_semirings ? S{T} : S
        fsm1 = FSM(
            [1 => one(K)],
            [(1, 1) => one(K), (1, 2) => one(K), (2, 2) => one(K),
             (2, 3) => one(K), (3, 3) => one(K)],
            [3 => one(K)],
            [Label(1), Label(2), Label(3)]
        )
        Z = one(K) + one(K)
        fsm2 = FSM(
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
    for S in weight_semirings, T in types
        K = S ∈ parametric_semirings ? S{T} : S
        fsm1 = FSM(
            [1 => one(K)],
            [(1, 1) => one(K), (1, 2) => one(K), (2, 2) => one(K),
             (2, 3) => one(K), (3, 3) => one(K)],
            [3 => one(K)],
            [Label(1), Label(2), Label(3)]
        )
        fsm2 = FSM(
            [1 => one(K)],
            [(1, 1) => one(K), (1, 2) => one(K),
             (2, 2) => one(K), (2, 1) => one(K)],
            [2 => one(K)],
            [Label(:a), Label(:b)]
        )
        fsm3 = FSM(
            [1 => one(K)],
            [(1, 1) => one(K), (1, 2) => one(K), (2, 2) => one(K),
             (2, 3) => one(K), (3, 3) => one(K),
             (4, 4) => one(K), (4, 5) => one(K), (5, 5) => one(K),
             (5, 6) => one(K), (6, 6) => one(K),
             (3, 1) => one(K), (3, 4) => one(K),
             (6, 1) => one(K), (6, 4) => one(K)],
            [6 => one(K)],
            [Label(:a) * Label(1), Label(:a) * Label(2),
             Label(:a) * Label(3), Label(:b) * Label(1),
             Label(:b) * Label(2), Label(:b) * Label(3)]
        )
        fsm = fsm2 ∘ [fsm1, fsm1]
        @test fsmequal(fsm, fsm3)
        @test length(nonzeros(fsm.α)) == length(nonzeros(fsm3.α))
        @test length(nonzeros(fsm.T)) == length(nonzeros(fsm3.T))
        @test length(nonzeros(fsm.ω)) == length(nonzeros(fsm3.ω))

        fsm = fsm2 ∘ Dict(Label(:a) => fsm1, Label(:b) => fsm1)
        @test fsmequal(fsm, fsm3)
        @test length(nonzeros(fsm.α)) == length(nonzeros(fsm3.α))
        @test length(nonzeros(fsm.T)) == length(nonzeros(fsm3.T))
        @test length(nonzeros(fsm.ω)) == length(nonzeros(fsm3.ω))
    end
end

@testset "propagate" begin
    for S in [LogSemiring, ProbSemiring], T in types
        K = S{T}
        v1, v2, v3 = one(K), one(K) + one(K), one(K) + one(K) + one(K)
        fsm1 = FSM(
            [1 => v2],
            [(1,2) => v1,
             (1,3) => v1,
             (2,4) => v1,
             (3,4) => v2],
            [4 => v1],
            [Label(:a), Label(:b), Label(:c), Label(:d)]
        )

        fsm2 = FSM(
            [1 => v2],
            [(1,2) => v2 * v1,
             (1,3) => v2 * v1,
             (2,4) => v2 * v1 * v1,
             (3,4) => v2 * v1 * v2],
            [4 => v2 * v1 * v1 + v2 * v1 * v2],
            [Label(:a), Label(:b), Label(:c), Label(:d)]
        )

        @test fsmequal(propagate(fsm1), fsm2)
        @test length(nonzeros(fsm1.α)) == length(nonzeros(fsm2.α))
        @test length(nonzeros(fsm1.T)) == length(nonzeros(fsm2.T))
        @test length(nonzeros(fsm1.ω)) == length(nonzeros(fsm2.ω))

        fsm = FSM([1 => one(K)], [], [1 => one(K)], [Label(1)])
        @test fsmequal(propagate(fsm), fsm)
    end
end


@testset "determinize" begin
    for S in divisible_semirings, T in types
        K = S ∈ parametric_semirings ? S{T} : S
        fsm1 = FSM(
            [1 => one(K), ],
            [(1,1) => one(K), (1, 2) => one(K), (1, 3) => one(K),
             (2, 4) => one(K), (3, 4) => one(K)],
            [4 => one(K)],
            [Label(:a), Label(:b), Label(:c), Label(:d)]
        )
        fsm1 = fsm1 ∪ fsm1
        fsm2 = determinize(fsm1)
        @test nstates(fsm2) < nstates(fsm1)
        @test fsmequal(fsm2 |> renorm, fsm1 |> renorm)

        cfsm1 = compose(fsm1, repeat([fsm1], nstates(fsm1)))
        cfsm2 = determinize(cfsm1)
        @test issetequal(Set(cfsm1.λ), Set(cfsm2.λ))
    end
end

@testset "minimize" begin
    for S in divisible_semirings, T in types
        K = S ∈ parametric_semirings ? S{T} : S
        fsm1 = FSM(
            [1 => one(K), ],
            [(1,1) => one(K), (1, 2) => one(K), (1, 3) => one(K),
             (2, 4) => one(K), (3, 4) => one(K)],
            [4 => one(K)],
            [Label(:a), Label(:b), Label(:c), Label(:d)]
        )
        fsm1 = fsm1 ∪ fsm1
        fsm2 = minimize(fsm1)
        @test nstates(fsm2) < nstates(fsm1)
        @test fsmequal(fsm2 |> renorm, fsm1 |> renorm)
    end
end

if CUDA.functional()
    @testset "cu adapt" begin
        for S in divisible_semirings, T in types
            K = S ∈ parametric_semirings ? S{T} : S
            fsm = FSM(
                [1 => one(K), ],
                [(1,1) => one(K), (1, 2) => one(K), (1, 3) => one(K),
                 (2, 4) => one(K), (3, 4) => one(K)],
                [4 => one(K)],
                [Label(:a), Label(:b), Label(:c), Label(:d)]
            )

            cufsm = adapt(CuArray, fsm)
            @test cufsm.α̂ isa CuSparseVector
            @test cufsm.T̂ isa CuSparseMatrix
            @test cufsm.λ isa Array
            @test all(fsm.α̂ .≈ SparseVector(cufsm.α̂))
            @test all(fsm.T̂ .≈ SparseMatrixCSC(cufsm.T̂))
         end
    end
end


