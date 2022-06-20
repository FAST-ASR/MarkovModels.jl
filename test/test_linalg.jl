
@testset "broadcast" begin
    pKs = [LogSemiring, ProbSemiring, TropicalSemiring]
    Ts = [Float32, Float64]
    for pK = pKs, T = Ts
        K = pK{T}
        sv = sparsevec([1, 3], K[0.1, 0.2], 3)
        dv = Array{K}(collect(1:3))
        cu_sv = adapt(CuArray, sv)
        cu_dv = adapt(CuArray, dv)

        for op in [*, /]
            r = broadcast!(op, similar(dv), sv, dv)
            cu_r = broadcast!(op, similar(cu_dv), cu_sv, cu_dv)
            @test all(r .≈ adapt(Array, cu_r))

            r = broadcast(op, sv, dv)
            cu_r = broadcast(op, cu_sv, cu_dv)
            @test all(r .≈ adapt(Array, cu_r))
        end
    end
end

@testset "sparse matrix multiplication" begin
    pKs = [LogSemiring, ProbSemiring, TropicalSemiring]
    Ts = [Float32, Float64]
    for pK = pKs, T = Ts
        K = pK{T}
        sm = sparse([1, 2, 2], [3, 2, 3], K[1, 2, 3], 3, 3)
        dm = reshape(Array{K}(collect(1:9)), 3, 3)
        dv = Array{K}(collect(1:3))
        cu_sm = adapt(CuArray, sm)
        cu_dm = adapt(CuArray, dm)
        cu_dv = adapt(CuArray, dv)

        r = mul!(similar(dm), sm, dm)
        cu_r = mul!(similar(cu_dm), cu_sm, cu_dm)
        @test all(r .≈ adapt(Array, cu_r))

        r = mul!(similar(dv), sm, dv)
        cu_r = mul!(similar(cu_dv), cu_sm, cu_dv)
        @test all(r .≈ adapt(Array, cu_r))
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

