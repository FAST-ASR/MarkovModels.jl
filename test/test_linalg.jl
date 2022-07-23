
@testset "vcat" begin
    pKs = [LogSemiring, ProbSemiring, TropicalSemiring]
    Ts = [Float32, Float64]
    for pK = pKs, T = Ts
        K = pK{T}
        sv = sparsevec([1, 3], K[0.1, 0.2], 3)
        cu_sv = adapt(CuArray, sv)

        r = vcat(sv, sv)
        cu_sv = vcat(cu_sv, cu_sv)
        @test cu_sv isa CuSparseVector
        @test all(r .≈ SparseVector(cu_sv))
    end
end

@testset "blockdiag" begin
    pKs = [LogSemiring, ProbSemiring, TropicalSemiring]
    Ts = [Float32, Float64]
    for pK = pKs, T = Ts
        K = pK{T}
        sm = sparse([1, 2, 2], [3, 2, 3], K[1, 2, 3], 3, 3)
        cu_sm_csc = adapt(CuArray, sm)
        cu_sm_csr = CuSparseMatrixCSR(cu_sm_csc)

        r = blockdiag(sm, sm, sm)
        cu_r_csc = blockdiag(cu_sm_csc, cu_sm_csc, cu_sm_csc)
        cu_r_csr = blockdiag(cu_sm_csr, cu_sm_csr, cu_sm_csr)
        @test all(r .≈ SparseMatrixCSC(cu_r_csc))
        @test all(r .≈ SparseMatrixCSC(cu_r_csr))
    end
end

@testset "broadcast" begin
    pKs = [LogSemiring, ProbSemiring, TropicalSemiring]
    Ts = [Float32, Float64]
    for pK = pKs, T = Ts
        K = pK{T}
        sv = sparsevec([1, 3], K[2, 3], 3)
        dv = Array{K}(collect(1:3))
        cu_sv = adapt(CuArray, sv)
        cu_dv = adapt(CuArray, dv)

        for op in [*, /]
            r = broadcast!(op, similar(dv), sv, dv)
            cu_r = broadcast!(op, similar(cu_dv), cu_sv, cu_dv)
            @test all(r .≈ Array(cu_r))

            r = broadcast(op, sv, dv)
            cu_r = broadcast(op, cu_sv, cu_dv)
            @test all(r .≈ Array(cu_r))
        end
    end
end

@testset "CSC/CSR conversion" begin
    pKs = [LogSemiring, ProbSemiring, TropicalSemiring]
    Ts = [Float32, Float64]
    for pK = pKs, T = Ts
        K = pK{T}
        sm = sparse([1, 2, 2, 3, 4], [3, 1, 2, 1, 3], K[1, 2, 3, 4, 5], 4, 3)
        cu_sm_csc = adapt(CuArray, sm)
        cu_sm_csr = CuSparseMatrixCSR(cu_sm_csc)
        cu_sm_csr_csc = CuSparseMatrixCSC(cu_sm_csr)


        @test all(SparseMatrixCSC(cu_sm_csr) .≈ SparseMatrixCSC(cu_sm_csc))
        @test all(SparseMatrixCSC(cu_sm_csr_csc) .≈ SparseMatrixCSC(cu_sm_csc))
    end
end

@testset "materialize adjoint/transpose" begin
    pKs = [LogSemiring, ProbSemiring, TropicalSemiring]
    Ts = [Float32, Float64]
    for pK = pKs, T = Ts
        K = pK{T}
        sm = sparse([1, 2, 2, 3, 4], [3, 1, 2, 1, 3], K[1, 2, 3, 4, 5], 4, 3)
        cu_sm_csc = adapt(CuArray, sm)
        cu_sm_csr = CuSparseMatrixCSR(cu_sm_csc)

        @test all(SparseMatrixCSC(copy(cu_sm_csc')) .≈ sm')
        @test all(SparseMatrixCSC(copy(cu_sm_csr')) .≈ sm')
        @test all(SparseMatrixCSC(copy(transpose(cu_sm_csc))) .≈ sm')
        @test all(SparseMatrixCSC(copy(transpose(cu_sm_csr))) .≈ sm')
    end
end

@testset "mul!" begin
    pKs = [LogSemiring, ProbSemiring, TropicalSemiring]
    Ts = [Float32, Float64]
    for pK = pKs, T = Ts
        K = pK{T}
        sm = sparse([1, 2, 2, 3, 4], [3, 1, 2, 1, 3], K[1, 2, 3, 4, 5], 4, 3)
        dm = reshape(Array{K}(collect(1:12)), 3, 4)
        dv = Array{K}(collect(1:3))
        cu_sm = CuSparseMatrixCSR(adapt(CuArray, sm))
        cu_dm = adapt(CuArray, dm)
        cu_dv = adapt(CuArray, dv)

        r = mul!(similar(dm, 4, 4), sm, dm)
        cu_r = mul!(similar(cu_dm, 4, 4), cu_sm, cu_dm)
        @test all(r .≈ Array(cu_r))

        r = mul!(similar(dv, 4), sm, dv)
        cu_r = mul!(similar(cu_dv, 4), cu_sm, cu_dv)
        @test all(r .≈ Array(cu_r))
    end
end

@testset "SparseLowRank" begin
    S_data = ([2, 1, 2, 3, 1, 2, 3], [1, 2, 2, 2, 3, 3, 3], [-0.8537556677220175, -0.5221596132743043, 0.0013913594071095746, 0.3591308843925371, 1.6900253084904349, 2.8440546099472055, 0.35142088167369034])
    D_data = ([1, 1, 2], [1, 2, 2], [0.11077377456243505, 0.09828006319199295, -0.8977197722280805])
    U_data = ([3, 1, 2], [1, 2, 2], [-0.09632806925224809, 0.1420214330245059, 1.382672876077084])
    V_data = ([1, 2, 3, 1, 3], [1, 1, 1, 2, 2], [-0.08086191422068456, 0.541004461949763, -0.5767570404556415, 1.7149073732620217, -1.0107885621908106])

    pKs = [LogSemiring, ProbSemiring, TropicalSemiring]
    Ts = [Float32, Float64]
    for pK = pKs, T = Ts
        K = pK{T}

        S = sparse(S_data[1], S_data[2], K.(S_data[3]))
        D = sparse(D_data[1], D_data[2], K.(D_data[3]))
        U = sparse(U_data[1], U_data[2], K.(U_data[3]))
        V = sparse(V_data[1], V_data[2], K.(V_data[3]))
        M = MarkovModels.SparseLowRankMatrix(S, D, U, V)

        @test all(M .≈ (S + U * (I + D) * V'))
        @test all(copy(M) .≈ (S + U * (I + D) * V'))
    end
end
