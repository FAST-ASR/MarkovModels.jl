# SPDX-License-Identifier: MIT

using CUDA, CUDA.CUSPARSE, SparseArrays
using MarkovModels
using Test

function makefsm(SF, S)
    fsm = FSM{SF}()

    prev = addstate!(fsm, pdfindex = 1)
    setinit!(prev)
    for s in 2:S
        state = addstate!(fsm, pdfindex = s)
        link!(fsm, prev, state)
        link!(fsm, state, state)
        prev = state
    end
    setfinal!(prev)
    renormalize!(fsm)
    fsm
end

const S = 10
const N = 20
const T = Float64
const SF = LogSemifield{T}

if CUDA.functional()
    @testset "CUDA sparse op" begin
        D = 5

        x = ones(T, D)
        y = spzeros(T, D)
        y[2] = SF(3)

        xg = CuArray(x)
        yg = CuSparseVector(y)
        out = fill!(similar(xg), zero(T))
        MarkovModels.elmul_svdv!(out, yg, xg)
        @test all(convert(Vector{T}, out) .≈ x .* y)

        Yᵀ = spdiagm(y)
        Yᵀg = CuSparseMatrixCSC(Yᵀ)
        Yg = CuSparseMatrixCSR( Yᵀg.colPtr, Yᵀg.rowVal, Yᵀg.nzVal, (D,D))
        out = fill!(similar(xg), zero(T))
        MarkovModels.mul_smdv!(out, Yg, xg)
        @test all(convert(Vector{T}, out) .≈ transpose(Yᵀ) * x)
    end
end

@testset "forward_backward" begin
    lhs = convert(Matrix{SF}, zeros(T, S, N))

    fsm = makefsm(SF, S)
    cfsm = compile(fsm, allocator = zeros)
    γ_dcpu, ttl_dcpu = αβrecursion(cfsm, lhs)

    cfsm = compile(fsm, allocator = spzeros)
    γ_scpu, ttl_scpu = @inferred αβrecursion(cfsm, lhs)
    @test all(convert(Matrix{T}, γ_dcpu) .≈ convert(Matrix{T}, γ_scpu))
    @test convert(T, ttl_dcpu) ≈ convert(T, ttl_scpu)

    if CUDA.functional()
        cfsm = compile(fsm, allocator = zeros) |> gpu
        γ_dgpu, ttl_dgpu = @inferred αβrecursion(cfsm, CuArray(lhs))
        @test all(convert(Matrix{T}, γ_dcpu) .≈ convert(Matrix{T}, γ_dgpu))
        @test convert(T, ttl_dcpu) ≈ convert(T, ttl_dgpu)

        cfsm = compile(fsm, allocator = spzeros) |> gpu
        γ_sgpu, ttl_sgpu = @inferred αβrecursion(cfsm, CuArray(lhs))
        @test all(convert(Matrix{T}, γ_dcpu) .≈ convert(Matrix{T}, γ_sgpu))
        @test convert(T, ttl_dcpu) ≈ convert(T, ttl_sgpu)
    end
end


