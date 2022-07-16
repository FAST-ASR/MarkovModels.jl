
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

#@testset "sv-dv" begin
#    sm = sprandn(1000, 1, 1e-5)
#    dv = randn(1000)

#    gt = similar(sm)
#    fill!(gt, zero(Float64))
#    broadcast!(*, gt, sm, dv)

#    @test elmul!(similar(sm), sm, dv) == gt
#    @test elmul!(similar(sm), dv, sm) == gt
#    @test elmul!(similar(sm'), sm', dv') == gt'
#    @test elmul!(similar(sm'), dv', sm') == gt'
#end

