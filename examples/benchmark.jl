# SPDX-License-Identifier: MIT

using ArgParse
using CUDA
using CUDA.CUSPARSE
using BenchmarkTools
using MarkovModels
using SparseArrays
using LinearAlgebra
using LogExpFunctions

BLAS.set_num_threads(1)

#######################################################################
# This is the "standard" way of implementing the forward-backward
# algorithm (as would be done in python for example).

function forward(Aᵀ, init, lhs::AbstractMatrix{T}) where T
    N = size(lhs, 2)
    log_α = fill!(similar(lhs), T(-Inf))
    log_α[:, 1] = view(lhs, :, 1) + init
    for n in 2:N
        log_α[:, n] = lhs[:, n] + logsumexp(Aᵀ .+ reshape(log_α[:,n-1], 1, :),
                                                     dims = 2)
    end
    log_α
end

function backward(A, final, lhs::AbstractMatrix{T}) where T
    N = size(lhs, 2)
    log_β = fill!(similar(lhs), T(-Inf))
    log_β[:, end] = final
    for n in N-1:-1:1
        tmp = dropdims(logsumexp(A .+ reshape(log_β[:,n+1] .+ lhs[:, n+1], 1, :),
                                 dims = 2), dims = 2)
        log_β[:, n] = tmp
    end
    log_β
end

function forward_backward(A, Aᵀ, init, final, lhs)
    log_α = forward(Aᵀ, init, lhs)
    log_β = backward(A, final, lhs)
    log_γ = log_α .+ log_β
    sums = logsumexp(log_γ, dims = 1)
    log_γ .- sums, minimum(sums)
end

#######################################################################

function makefsm(SF, S)
    fsm = FSM{SF}()

    prev = addstate!(fsm, pdfindex = 1)
    setinit!(prev)
    link!(fsm, prev, prev)
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

function main(T, SF, S, N)
    # Generate some pseudo-(log)likelihood
    lhs = convert(Matrix{SF}, zeros(T, S, N))

    # Build the Finite State Machine (FSM).
    fsm = makefsm(SF, S)

    println("Setup:")
    println("  float type: $T")
    println("  # states: $S")
    println("  # frames: $N")
    println()

    println("αβrecursion (standard implementation) with dense CPU arrays:")
    cfsm = compile(fsm, allocator = zeros)
    Aᵀ = convert(Matrix{T}, cfsm.Aᵀ)
    A = convert(Matrix{T}, cfsm.A)
    init = convert(Vector{T}, cfsm.π)
    final = convert(Vector{T}, cfsm.ω)
    lhs_rs = convert(Matrix{T}, lhs)
    @btime forward_backward($A, $Aᵀ, $init, $final, $lhs_rs)
    println("------------------------------------------------")

    println("αβrecursion with dense CPU arrays:")
    cfsm = compile(fsm, allocator = zeros)
    @btime αβrecursion($cfsm, $lhs)
    println("------------------------------------------------")

    println("αβrecursion with sparse CPU arrays:")
    cfsm = compile(fsm, allocator = spzeros)
    @btime αβrecursion($cfsm, $lhs)
    println("------------------------------------------------")

    println("αβrecursion (standard implementation) with dense GPU arrays:")
    cfsm = compile(fsm, allocator = zeros)
    Aᵀ = convert(Matrix{T}, cfsm.Aᵀ) |> CuArray
    A = convert(Matrix{T}, cfsm.A) |> CuArray
    init = convert(Vector{T}, cfsm.π) |> CuArray
    final = convert(Vector{T}, cfsm.ω) |> CuArray
    lhs_rs = convert(Matrix{T}, lhs) |> CuArray
    @btime forward_backward($A, $Aᵀ, $init, $final, $lhs_rs)
    println("------------------------------------------------")

    println("αβrecursion with dense GPU arrays:")
    cfsm = compile(fsm, allocator = zeros) |> gpu
    lhs = CuArray(lhs)
    @btime αβrecursion($cfsm, $lhs)
    println("------------------------------------------------")

    println("αβrecursion with sparse GPU arrays:")
    cfsm = compile(fsm, allocator = spzeros) |> gpu
    @btime αβrecursion($cfsm, $lhs)
    println("------------------------------------------------")
end

s = ArgParseSettings()
@add_arg_table s begin
    "--num-frames", "-N"
        help = "number of observation frames"
        arg_type = Int
        default = 1000
    "--num-states", "-S"
        help = "number of states"
        arg_type = Int
        default = 1000
    "--single-precision"
        help = "use single precision"
        action = :store_true
end

args = parse_args(s)

const T = args["single-precision"] ? Float32 : Float64
const SF = LogSemifield{T}
const S = args["num-states"]
const N = args["num-frames"]
main(T, SF, S, N)
