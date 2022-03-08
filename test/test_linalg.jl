import MarkovModels.Inference: elmul!

@testset "element-wise mult" begin
    @testset "sm-dv" begin
        sm = sprandn(1000, 1000, 1e-2)
        dv = randn(1000)

        gt = similar(sm)
        fill!(gt, zero(Float64))
        broadcast!(*, gt, sm, dv)

        gtT = similar(sm)
        fill!(gtT, zero(Float64))
        broadcast!(*, gtT, sm, dv')

        @test elmul!(similar(sm), sm, dv) == gt
        @test elmul!(similar(sm), dv, sm) == gt
        @test elmul!(similar(sm), sm, dv') == gtT
        @test elmul!(similar(sm), dv', sm) == gtT
        @test elmul!(similar(sm), sm', dv') == gt'
        @test elmul!(similar(sm), sm', dv) == gtT'
    end

    @testset "sm-dm" begin
        sm = sprandn(1000, 1000, 1e-3)
        dm = randn(1000, 1000)

        gt = similar(sm)
        fill!(gt, zero(Float64))
        broadcast!(*, gt, sm, dm)

        @test elmul!(similar(sm), sm, dm) == gt
        @test elmul!(similar(sm), dm, sm) == gt
        @test elmul!(similar(sm), sm', dm') == gt'
        @test elmul!(similar(sm), dm', sm') == gt'
    end

    @testset "sv-dv" begin
        sm = sprandn(1000, 1, 1e-5)
        dv = randn(1000)

        gt = similar(sm)
        fill!(gt, zero(Float64))
        broadcast!(*, gt, sm, dv)

        @test elmul!(similar(sm), sm, dv) == gt
        @test elmul!(similar(sm), dv, sm) == gt
        @test elmul!(similar(sm'), sm', dv') == gt'
        @test elmul!(similar(sm'), dv', sm') == gt'
    end

end
