# SPDX-License-Identifier: MIT

using CUDA, CUDA.CUSPARSE, SparseArrays
import LogExpFunctions: logsumexp
using MarkovModels
import MarkovModels: logaddexp
using Test

const D = 5 # vector/matrix dimension
const B = 2 # batch size
const S = 3 # number of states
const K = 3 # Number of emissions
const N = 5 # number of frames

const T = Float64
const SF = LogSemifield{T}


function makefsm(SF, S)
    fsm = FSM{SF}()

    prev = addstate!(fsm, pdfindex = 1)
    setinit!(prev)
    addarc!(fsm, prev, prev)
    for s in 2:S
        state = addstate!(fsm, pdfindex = s)
        addarc!(fsm, prev, state)
        addarc!(fsm, state, state)
        prev = state
    end
    setfinal!(prev)
    renormalize!(fsm)
    fsm
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

function forward_backward(A, Aᵀ, init, final, lhs)
    log_α = forward(A, init, lhs)
    log_β = backward(Aᵀ, final, lhs)
    log_γ = log_α .+ log_β
    sums = logsumexp(log_γ, dims = 1)
    exp.(log_γ .- sums), minimum(sums)
end


@testset "logaddexp" begin
    @test (@inferred logaddexp(2.0, 3.0)) ≈ log(exp(2.0) + exp(3.0))
    @test (@inferred logaddexp(10002.0, 10003.0)) ≈ 10000 + logaddexp(2.0, 3.0)
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
        MarkovModels.elmul_svdv!(out, yg, xg)
        @test all(convert(Vector{T}, out) .≈ x .* y)

        x = ones(T, D)
        y = spzeros(T, D)
        xg = CuArray(x)
        yg = CuSparseVector(y)
        out = similar(xg)
        MarkovModels.elmul_svdv!(out, yg, xg)
        @test all(convert(Vector{T}, out) .≈ x .* y)

        for Yᵀ in [spdiagm(y), spzeros(T, length(y))]
            Yᵀ = spdiagm(y)
            Yᵀg = CuSparseMatrixCSC(Yᵀ)
            Yg = CuSparseMatrixCSR( Yᵀg.colPtr, Yᵀg.rowVal, Yᵀg.nzVal, (D,D))
            out = similar(xg)
            MarkovModels.mul_smdv!(out, Yg, xg)
            @test all(convert(Vector{T}, out) .≈ transpose(Yᵀ) * x)
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
        MarkovModels.elmul_svdm!(out, yg, Xg)
        @test all(convert(Matrix{T}, out) .≈ X .* reshape(y, :, 1))

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
        MarkovModels.elmul_svdm!(out, yg, Xg)
        @test all(convert(Matrix{T}, out) .≈ X .* reshape(y, :, 1))

        A = spzeros(3, 3)
        Ag = CuSparseMatrixCSC(A)
        Ag = CuSparseMatrixCSR(Ag.colPtr, Ag.rowVal, Ag.nzVal, Ag.dims)
        Y = ones(3, 3)
        Yg = CuArray(Y)
        out = similar(Yg)
        MarkovModels.mul_smdm!(out, Ag, Yg)
        @test all(convert(Matrix{T}, out) .≈ A' * Y)

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
        MarkovModels.mul_smdm!(out, Ag, Yg)
        @test all(convert(Matrix{T}, out) .≈ A' * Y)

        A = ones(T, 3, 2)
        Ag = CuArray(A)
        B = ones(T, 2, 3)
        Bg = CuArray(B)
        out = similar(Ag, 3, 3)
        MarkovModels.mul_dmdm!(out, Ag, Bg)
        @test all(convert(Matrix{T}, out) .≈ A * B)
    end
end

@testset "forward_backward" begin
    lhs = ones(T, S, N)

    fsm = makefsm(SF, S)
    cfsm = compile(fsm, K)

    γ_ref, ttl_ref = forward_backward(
        convert(Matrix{T}, cfsm.T),
        convert(Matrix{T}, cfsm.Tᵀ),
        convert(Vector{T}, cfsm.π),
        convert(Vector{T}, cfsm.ω),
        convert(Matrix{T}, lhs)
    )

    γ_scpu, ttl_scpu = pdfposteriors(cfsm, lhs)
    @test all(γ_ref .≈ γ_scpu)
    @test ttl_ref ≈ ttl_scpu

    if CUDA.functional()
        cfsm = cfsm |> gpu
        γ_sgpu, ttl_sgpu = pdfposteriors(cfsm, CuArray(lhs))
        @test all(γ_ref .≈ convert(Matrix{T}, γ_sgpu))
        @test ttl_ref ≈ ttl_sgpu
    end
end

@testset "batch forward_backward" begin
    lhs = ones(T, S, N, B)

    fsm = makefsm(SF, S)
    cfsm = compile(fsm, K)

    γ_ref, ttl_ref = forward_backward(
        convert(Matrix{T}, cfsm.T),
        convert(Matrix{T}, cfsm.Tᵀ),
        convert(Vector{T}, cfsm.π),
        convert(Vector{T}, cfsm.ω),
        convert(Matrix{T}, lhs[:, :, 1])
    )

    cfsms = union([cfsm for i in 1:B]...)
    γ_scpu, ttl_scpu = pdfposteriors(cfsms, lhs)
    for b in 1:B
        @test all(γ_ref .≈ γ_scpu[:,:,b])
        @test ttl_ref ≈ ttl_scpu[b]
    end

    if CUDA.functional()
        cfsms = cfsms |> gpu
        γ_sgpu, ttl_sgpu = pdfposteriors(cfsms, CuArray(lhs))
        for b in 1:B
            @test all(γ_ref .≈ convert(Array{T,3}, γ_sgpu)[:,:,b])
            @test ttl_ref ≈ convert(Vector{T}, ttl_sgpu)[b]
        end
    end
end

@testset "remove_eps" begin
    SF = LogSemifield{Float64}
    fsm = FSM{SF}()

    s1 = addstate!(fsm, label = "a", pdfindex = 1)
    s2 = addstate!(fsm)
    s3 = addstate!(fsm, label = "c", pdfindex = 2)
    setinit!(s1)
    setfinal!(s3)

    addarc!(fsm, s1, s2)
    addarc!(fsm, s2, s3)
    addarc!(fsm, s3, s1)

    ns = collect(filter(s -> ! MarkovModels.isemitting(s),
                        MarkovModels.states(fsm |> remove_eps)))
    @test length(ns) == 0

    setinit!(s2)
    @test_throws MarkovModels.InvalidFSMError remove_eps(fsm)

    setinit!(s2, zero(SF))
    setfinal!(s2)
    @test_throws MarkovModels.InvalidFSMError remove_eps(fsm)

    setfinal!(s2, zero(SF))
    s2.label = "b"
    @test_throws MarkovModels.InvalidFSMError remove_eps(fsm)
end

@testset "determinize" begin
    fsm = FSM()

    s1 = addstate!(fsm, label = "a", pdfindex = 1)
    s2 = addstate!(fsm)
    s3 = addstate!(fsm, label = "c", pdfindex = 2)
    setinit!(s1)
    setfinal!(s3)

    addarc!(fsm, s1, s2)
    addarc!(fsm, s2, s3)
    addarc!(fsm, s3, s1)

    @test_throws MarkovModels.InvalidFSMError determinize(fsm)
end

