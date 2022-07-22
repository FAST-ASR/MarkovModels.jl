# SPDX-License-Identifier: MIT

const weight_semirings = [BoolSemiring, LogSemiring, ProbSemiring,
                          TropicalSemiring]
const parametric_semirings = Set([LogSemiring, ProbSemiring, TropicalSemiring])
const divisible_semirings = Set([LogSemiring, ProbSemiring, TropicalSemiring])
const types = [Float32, Float64]

function fsaequal(fsa1::FSA, fsa2::FSA)
    n = max(nstates(fsa1), nstates(fsa2))
    ls1 = totallabelsum(fsa1, n)
    ls2 = totallabelsum(fsa2, n)
    ws1 = totalweightsum(fsa1, n)
    ws2 = totalweightsum(fsa2, n)
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
        fsa1 = FSA(α, T, ω, [Label(1), Label(2)])
        fsa2 = FSA(
            sparsevec([1], [one(K)], 2),
            sparse([1, 1, 2, 2], [1, 2, 2, 1], ones(K, 4), 2, 2),
            sparsevec([2], [one(K)], 2),
            [Label(1), Label(2)]
        )
        @test fsaequal(fsa1, fsa2)

        onestr = "$(val(one(K)))"
        fsa3 = FSA("""
        {
            "semiring": "$K",
            "initstates": [[1, $onestr]],
            "arcs": [[1, 1, $onestr], [1, 2, $onestr], [2, 2, $onestr],
                     [2, 1, $onestr]],
            "finalstates": [[2, $onestr]],
            "labels": [1, 2]
        }
        """)
        @test fsaequal(fsa1, fsa3)

        α = [1 => one(K)]
        T = []
        ω = [1 => one(K)]
        fsa1 = FSA(α, T, ω, [Label(1)])
        fsa2 = FSA(
            sparsevec([1], [one(K)], 1),
            spzeros(K, 1, 1),
            sparsevec([1], [one(K)], 1),
            [Label(1)]
        )
        @test fsaequal(fsa1, fsa2)
    end
end

@testset "union" begin
    for S in weight_semirings, T in types
        K = S ∈ parametric_semirings ? S{T} : S
        fsa1 = FSA(
            [1 => one(K)],
            [(1, 1) => one(K), (1, 2) => one(K), (2, 2) => one(K),
             (2, 3) => one(K), (3, 3) => one(K)],
            [3 => one(K)],
            [Label(1), Label(2), Label(3)]
        )
        fsa2 = FSA(
            [1 => one(K)],
            [(1, 1) => one(K), (1, 2) => one(K), (2, 2) => one(K),
             (2, 3) => one(K), (3, 3) => one(K)],
            [3 => one(K)],
            [Label(4), Label(5), Label(6)]
        )
        fsa3 = FSA(
            [1 => one(K), 4 => one(K)],
            [(1, 1) => one(K), (1, 2) => one(K), (2, 2) => one(K),
             (2, 3) => one(K), (3, 3) => one(K),
             (4, 4) => one(K), (4, 5) => one(K), (5, 5) => one(K),
             (5, 6) => one(K), (6, 6) => one(K)],
            [3 => one(K), 6 => one(K)],
            [Label(1), Label(2), Label(3), Label(4), Label(5), Label(6)]
        )
        fsa = union(fsa1, fsa2)
        @test fsaequal(fsa, fsa3)
        @test length(nonzeros(fsa.α)) == length(nonzeros(fsa3.α))
        @test length(nonzeros(fsa.T)) == length(nonzeros(fsa3.T))
        @test length(nonzeros(fsa.ω)) == length(nonzeros(fsa3.ω))
    end
end

@testset "concatenation" begin
    for S in weight_semirings, T in types
        K = S ∈ parametric_semirings ? S{T} : S
        fsa1 = FSA(
            [1 => one(K)],
            [(1, 1) => one(K), (1, 2) => one(K), (2, 2) => one(K),
             (2, 3) => one(K), (3, 3) => one(K)],
            [3 => one(K)],
            [Label(1), Label(2), Label(3)]
        )
        fsa2 = FSA(
            [1 => one(K)],
            [(1, 1) => one(K), (1, 2) => one(K), (2, 2) => one(K),
             (2, 3) => one(K), (3, 3) => one(K)],
            [3 => one(K)],
            [Label(4), Label(5), Label(6)]
        )
        fsa3 = FSA(
            [1 => one(K)],
            [(1, 1) => one(K), (1, 2) => one(K), (2, 2) => one(K),
             (2, 3) => one(K), (3, 3) => one(K), (3, 4) => one(K),
             (4, 4) => one(K), (4, 5) => one(K), (5, 5) => one(K),
             (5, 6) => one(K), (6, 6) => one(K)],
            [6 => one(K)],
            [Label(1), Label(2), Label(3), Label(4), Label(5), Label(6)]
        )
        fsa = cat(fsa1, fsa2)
        @test fsaequal(fsa, fsa3)
        @test length(nonzeros(fsa.α)) == length(nonzeros(fsa3.α))
        @test length(nonzeros(fsa.T)) == length(nonzeros(fsa3.T))
        @test length(nonzeros(fsa.ω)) == length(nonzeros(fsa3.ω))
    end
end

@testset "renormalize" begin
    for S in divisible_semirings, T in types
        K = S ∈ parametric_semirings ? S{T} : S
        fsa1 = FSA(
            [1 => one(K) + one(K)],
            [(1, 1) => one(K), (1, 2) => one(K), (2, 2) => one(K),
             (2, 3) => one(K), (3, 3) => one(K)],
            [3 => one(K)],
            [Label(1), Label(2), Label(3)]
        )
        Z = one(K) + one(K)
        fsa2 = FSA(
            [1 => one(K)],
            [(1, 1) => one(K) / Z,
             (1, 2) => one(K) / Z,
             (2, 2) => one(K) / Z,
             (2, 3) => one(K) / Z,
             (3, 3) => one(K) / Z],
            [3 => one(K) / Z],
            [Label(1), Label(2), Label(3)]
        )
        fsa = renorm(fsa1)
        @test fsaequal(fsa, fsa2)
        @test length(nonzeros(fsa.α)) == length(nonzeros(fsa2.α))
        @test length(nonzeros(fsa.T)) == length(nonzeros(fsa2.T))
        @test length(nonzeros(fsa.ω)) == length(nonzeros(fsa2.ω))
    end
end

@testset "reversal" begin
    for S in weight_semirings, T in types
        K = S ∈ parametric_semirings ? S{T} : S
        fsa1 = FSA(
            [1 => one(K)],
            [(1, 1) => one(K), (1, 2) => one(K), (2, 2) => one(K),
             (2, 3) => one(K), (3, 3) => one(K)],
            [3 => one(K)],
            [Label(1), Label(2), Label(3)]
        )
        Z = one(K) + one(K)
        fsa2 = FSA(
            [3 => one(K)],
            [(1, 1) => one(K),
             (2, 1) => one(K),
             (2, 2) => one(K),
             (3, 2) => one(K),
             (3, 3) => one(K) ],
            [1 => one(K)],
            [Label(1), Label(2), Label(3)]
        )
        fsa = fsa1'
        @test fsaequal(fsa, fsa2)
        @test fsaequal(fsa', fsa1)
    end
end

@testset "replace" begin
    for S in weight_semirings, T in types
        K = S ∈ parametric_semirings ? S{T} : S
        fsa1 = FSA(
            [1 => one(K)],
            [(1, 1) => one(K), (1, 2) => one(K), (2, 2) => one(K),
             (2, 3) => one(K), (3, 3) => one(K)],
            [3 => one(K)],
            [Label(1), Label(2), Label(3)]
        )
        fsa2 = FSA(
            [1 => one(K)],
            [(1, 1) => one(K), (1, 2) => one(K),
             (2, 2) => one(K), (2, 1) => one(K)],
            [2 => one(K)],
            [Label(:a), Label(:b)]
        )
        fsa3 = FSA(
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
        fsa = replace(fsa2, [fsa1, fsa1])
        @test fsaequal(fsa, fsa3)
        @test length(nonzeros(fsa.α)) == length(nonzeros(fsa3.α))
        @test length(nonzeros(fsa.T)) == length(nonzeros(fsa3.T))
        @test length(nonzeros(fsa.ω)) == length(nonzeros(fsa3.ω))

        dict = Dict(:a => fsa1, :b => fsa1)
        fsa = replace(fsa2) do i
            dict[val(fsa2.λ[i])[end]]
        end
        @test fsaequal(fsa, fsa3)
        @test length(nonzeros(fsa.α)) == length(nonzeros(fsa3.α))
        @test length(nonzeros(fsa.T)) == length(nonzeros(fsa3.T))
        @test length(nonzeros(fsa.ω)) == length(nonzeros(fsa3.ω))
    end
end

@testset "propagate" begin
    for S in [LogSemiring, ProbSemiring], T in types
        K = S{T}
        v1, v2, v3 = one(K), one(K) + one(K), one(K) + one(K) + one(K)
        fsa1 = FSA(
            [1 => v2],
            [(1,2) => v1,
             (1,3) => v1,
             (2,4) => v1,
             (3,4) => v2],
            [4 => v1],
            [Label(:a), Label(:b), Label(:c), Label(:d)]
        )

        fsa2 = FSA(
            [1 => v2],
            [(1,2) => v2 * v1,
             (1,3) => v2 * v1,
             (2,4) => v2 * v1 * v1,
             (3,4) => v2 * v1 * v2],
            [4 => v2 * v1 * v1 + v2 * v1 * v2],
            [Label(:a), Label(:b), Label(:c), Label(:d)]
        )

        @test fsaequal(propagate(fsa1), fsa2)
        @test length(nonzeros(fsa1.α)) == length(nonzeros(fsa2.α))
        @test length(nonzeros(fsa1.T)) == length(nonzeros(fsa2.T))
        @test length(nonzeros(fsa1.ω)) == length(nonzeros(fsa2.ω))

        fsa = FSA([1 => one(K)], [], [1 => one(K)], [Label(1)])
        @test fsaequal(propagate(fsa), fsa)
    end
end


@testset "determinize" begin
    for S in divisible_semirings, T in types
        K = S ∈ parametric_semirings ? S{T} : S
        fsa1 = FSA(
            [1 => one(K), ],
            [(1,1) => one(K), (1, 2) => one(K), (1, 3) => one(K),
             (2, 4) => one(K), (3, 4) => one(K)],
            [4 => one(K)],
            [Label(:a), Label(:b), Label(:c), Label(:d)]
        )
        fsa1 = fsa1 ∪ fsa1
        fsa2 = determinize(fsa1)
        @test nstates(fsa2) < nstates(fsa1)
        @test fsaequal(fsa2 |> renorm, fsa1 |> renorm)

        cfsa1 = replace(fsa1, repeat([fsa1], nstates(fsa1)))
        cfsa2 = determinize(cfsa1)
        @test issetequal(Set(cfsa1.λ), Set(cfsa2.λ))
    end
end

@testset "minimize" begin
    for S in divisible_semirings, T in types
        K = S ∈ parametric_semirings ? S{T} : S
        fsa1 = FSA(
            [1 => one(K), ],
            [(1,1) => one(K), (1, 2) => one(K), (1, 3) => one(K),
             (2, 4) => one(K), (3, 4) => one(K)],
            [4 => one(K)],
            [Label(:a), Label(:b), Label(:c), Label(:d)]
        )
        fsa1 = fsa1 ∪ fsa1
        fsa2 = minimize(fsa1)
        @test nstates(fsa2) < nstates(fsa1)
        @test fsaequal(fsa2 |> renorm, fsa1 |> renorm)
    end
end

if CUDA.functional()
    @testset "cu adapt" begin
        for S in divisible_semirings, T in types
            K = S ∈ parametric_semirings ? S{T} : S
            fsa = FSA(
                [1 => one(K), ],
                [(1,1) => one(K), (1, 2) => one(K), (1, 3) => one(K),
                 (2, 4) => one(K), (3, 4) => one(K)],
                [4 => one(K)],
                [Label(:a), Label(:b), Label(:c), Label(:d)]
            )

            cufsa = adapt(CuArray, fsa)
            @test cufsa.α̂ isa CuSparseVector
            @test cufsa.T̂ isa CuSparseMatrix
            @test cufsa.λ isa Array
            @test all(fsa.α̂ .≈ SparseVector(cufsa.α̂))
            @test all(fsa.T̂ .≈ SparseMatrixCSC(cufsa.T̂))
         end
    end
end


