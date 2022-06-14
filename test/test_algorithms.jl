# SPDX-License-Identifier: MIT

const D = 5 # vector/matrix dimension
const B = 2 # batch size
const S = 3 # number of states
const K = 3 # Number of emissions
const N = 7 # number of frames

const T = Float32
const SF = LogSemiring{T}


function makefsm(SF, S)
    fsm = VectorFSM{SF}()

    prev = addstate!(fsm, 1, initweight = one(SF))
    addarc!(fsm, prev, prev)
    for s in 2:S
        fw = s == S ? one(SF) : zero(SF)
        state = addstate!(fsm, s, finalweight = fw)
        addarc!(fsm, prev, state)
        addarc!(fsm, state, state)
        prev = state
    end
    fsm |> renormalize
end

function forward(A, init, lhs::AbstractMatrix{T}) where T
    N = size(lhs, 2)
    log_α = similar(lhs)
    log_α[:, 1] = lhs[:, 1] + init
    for n in 2:N
        log_α[:, n] = lhs[:,n] + dropdims(logsumexp(A .+ log_α[:,n-1], dims = 1),
                                          dims = 1)
    end
    log_α
end

function backward(Aᵀ, final, lhs::AbstractMatrix{T}) where T
    N = size(lhs, 2)
    log_β = similar(lhs)
    log_β[:, end] = final
    for n in N-1:-1:1
        log_β[:, n] = dropdims(logsumexp(Aᵀ .+ log_β[:,n+1] .+ lhs[:,n+1], dims = 1),
                               dims = 1)
    end
    log_β
end

function forward_backward(A, Aᵀ, init, lhs)
    final = A[1:end-1,end]

    # ignore the final state
    A = A[1:end-1,1:end-1]
    Aᵀ = Aᵀ[1:end-1,1:end-1]
    init = init[1:end-1]

    log_α = forward(A, init, lhs)
    log_β = backward(Aᵀ, final, lhs)
    log_γ = log_α .+ log_β
    sums = logsumexp(log_γ, dims = 1)
    exp.(log_γ .- sums), minimum(sums)
end


if CUDA.functional()
    @testset "linalg" begin
        D = 5

        x = ones(T, D)
        y = spzeros(T, D)
        y[2] = T(3)
        xg = CuArray(x)
        yg = CuSparseVector(y)
        out = similar(xg)
        MarkovModels.Inference.elmul_svdv!(out, yg, xg)
        @test all(val.(out) .≈ x .* y)

        x = ones(T, D)
        y = spzeros(T, D)
        xg = CuArray(x)
        yg = CuSparseVector(y)
        out = similar(xg)
        MarkovModels.Inference.elmul_svdv!(out, yg, xg)
        @test all(val.(out) .≈ x .* y)

        for Yᵀ in [spdiagm(y), spzeros(T, length(y))]
            Yᵀ = spdiagm(y)
            Yᵀg = CuSparseMatrixCSC(Yᵀ)
            Yg = CuSparseMatrixCSR( Yᵀg.colPtr, Yᵀg.rowVal, Yᵀg.nzVal, (D,D))
            out = similar(xg)
            MarkovModels.Inference.mul_smdv!(out, Yg, xg)
            @test all(val.(out) .≈ transpose(Yᵀ) * x)
        end

        x = ones(T, D)
        y = spzeros(T, D)
        X = ones(T, D, D)
        Y = spzeros(T, D, D)
        xg = CuArray(x)
        yg = CuSparseVector(y)
        Xg = CuArray(X)
        Yg = CuSparseMatrixCSC(Y)
        Yg = CuSparseMatrixCSR(Yg.colPtr, Yg.rowVal, Yg.nzVal, Yg.dims)
        out = similar(Xg)
        MarkovModels.Inference.elmul_svdm!(out, yg, Xg)
        @test all(val.(out) .≈ X .* reshape(y, :, 1))

        x = ones(T, D)
        y = spzeros(T, D)
        X = ones(T, D, D)
        Y = spzeros(T, D, D)
        Y[1,1] = T(1)
        Y[1,2] = T(2)
        Y[2,1] = T(3)
        xg = CuArray(x)
        yg = CuSparseVector(y)
        Xg = CuArray(X)
        Yg = CuSparseMatrixCSC(Y)
        Yg = CuSparseMatrixCSR(Yg.colPtr, Yg.rowVal, Yg.nzVal, Yg.dims)
        out = similar(Xg)
        MarkovModels.Inference.elmul_svdm!(out, yg, Xg)
        @test all(val.(out) .≈ X .* reshape(y, :, 1))

        A = spzeros(3, 3)
        Ag = CuSparseMatrixCSC(A)
        Ag = CuSparseMatrixCSR(Ag.colPtr, Ag.rowVal, Ag.nzVal, Ag.dims)
        Y = ones(3, 3)
        Yg = CuArray(Y)
        out = similar(Yg)
        MarkovModels.Inference.mul_smdm!(out, Ag, Yg)
        @test all(val.(out) .≈ A' * Y)

        A = spzeros(3, 3)
        A[1, 1] = 1/2
        A[1, 2] = 1/2
        A[2, 2] = 1/2
        A[2, 3] = 1/2
        A[3, 3] = 1/2
        Ag = CuSparseMatrixCSC(A)
        Ag = CuSparseMatrixCSR(Ag.colPtr, Ag.rowVal, Ag.nzVal, Ag.dims)
        Y = ones(3, 3)
        Yg = CuArray(Y)
        out = similar(Yg)
        MarkovModels.Inference.mul_smdm!(out, Ag, Yg)
        @test all(val.(out) .≈ A' * Y)

        A = ones(T, 3, 2)
        Ag = CuArray(A)
        B = ones(T, 2, 3)
        Bg = CuArray(B)
        out = similar(Ag, 3, 3)
        MarkovModels.Inference.mul_dmdm!(out, Ag, Bg)
        @test all(val.(out) .≈ A * B)
    end

    @testset "array" begin
        X₁ = spzeros(T, 4, 3)
        X₂ = spzeros(T, 3, 2)

        X₁[1, 1] = T(2)
        X₁[3, 3] = T(3)
        X₁[4, 2] = T(-1)

        X₂[2, 1] = T(2)
        X₂[2, 2] = T(3)

        Y = blockdiag(X₁, X₂)

        gX₁ = CuSparseMatrixCSR(cu(X₁))
        gX₂ = CuSparseMatrixCSR(cu(X₂))
        gY = blockdiag(gX₁, gX₂)

        @test all(val.(gY) .≈ val.(Y))

        x₁ = spzeros(T, 3)
        x₂ = spzeros(T, 2)

        x₁[2] = T(1)
        x₂[1] = T(2)
        y = vcat(x₁, x₂)

        gx₁ = cu(x₁)
        gx₂ = cu(x₂)
        gy = vcat(gx₁, gx₂)

        gy = CUDA.@allowscalar Array(gy)
        @test all(gy .≈ y)
    end
end

@testset "forward_backward" begin
    lhs = ones(T, S, N)
    label2pdfid = Dict(1 => 1, 2 => 2, 3 => 3)

    fsm = makefsm(SF, S)
    mfsm = MatrixFSM(fsm, label2pdfid)

    γ_ref, ttl_ref = forward_backward(
        val.(mfsm.T),
        val.(mfsm.Tᵀ),
        val.(mfsm.π),
        lhs
    )

    γ_scpu, ttl_scpu = pdfposteriors(mfsm, lhs)
    @test all(γ_ref .≈ γ_scpu)
    @test ttl_ref ≈ ttl_scpu

    if CUDA.functional()
        mfsm = mfsm |> gpu
        γ_sgpu, ttl_sgpu = pdfposteriors(mfsm, CuArray(lhs))
        @test all(γ_ref .≈ val.(γ_sgpu))
        @test ttl_ref ≈ ttl_sgpu
    end
end

@testset "batch forward_backward" begin
    lhs = ones(T, S, 7, 2)
    seqlengths = [5, 7]

    label2pdfid = Dict(1 => 1, 2 => 2, 3 => 3)

    fsm = makefsm(SF, S)
    mfsm = MatrixFSM(fsm, label2pdfid)

    γ_ref1, ttl_ref1 = forward_backward(
        val.(mfsm.T),
        val.(mfsm.Tᵀ),
        val.(mfsm.π),
        lhs[:, 1:5, 1]
    )

    γ_ref2, ttl_ref2 = forward_backward(
        val.(mfsm.T),
        val.(mfsm.Tᵀ),
        val.(mfsm.π),
        lhs[:, 1:7, 1]
    )


    mfsms = union([mfsm for i in 1:B]...)
    γ_scpu, ttl_scpu = pdfposteriors(mfsms, lhs, seqlengths)
    @test all(γ_ref1 .≈ γ_scpu[:,1:5,1])
    @test ttl_ref1 ≈ ttl_scpu[1]
    @test all(γ_ref2 .≈ γ_scpu[:,1:7,2])
    @test ttl_ref2 ≈ ttl_scpu[2]
    @test all(γ_scpu[:,6:7,1] .== zero(T))

    if CUDA.functional()
        mfsm = mfsm |> gpu
        mfsms = union([mfsm for i in 1:B]...)
        γ_sgpu, ttl_sgpu = pdfposteriors(mfsms, CuArray(lhs), seqlengths)
        @test all(γ_ref1 .≈ val.(γ_sgpu)[:,1:5,1])
        @test ttl_ref1 ≈ val.(ttl_sgpu)[1]
        @test all(γ_ref2 .≈ val.(γ_sgpu)[:,1:7,2])
        @test ttl_ref2 ≈ val.(ttl_sgpu)[2]
        @test all(val.(γ_sgpu)[:,6:7,1] .== zero(T))
    end
end

@testset "bestpath" begin
    fsm = VectorFSM{SF}()

    s1 = addstate!(fsm, "a", initweight = one(SF))
    s2 = addstate!(fsm, "b")
    s3 = addstate!(fsm, "c")
    s4 = addstate!(fsm, "d", finalweight = one(SF))

    addarc!(fsm, s1, s2)
    addarc!(fsm, s2, s3)
    addarc!(fsm, s3, s4)

    lhs = ones(T, 4, 4)

    label2pdfid = Dict("a" => 1, "b" => 2, "c" => 3, "d" => 4)

    mfsm = MatrixFSM(fsm, label2pdfid)
    mfsm = convert(MatrixFSM{TropicalSemiring{T}}, mfsm)
    μ = maxstateposteriors(mfsm, lhs)
    path = bestpath(mfsm, μ)

    @test join(mfsm.labels[path], " ") == "a b c d"
end

