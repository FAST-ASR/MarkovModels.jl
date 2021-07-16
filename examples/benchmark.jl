# SPDX-License-Identifier: MIT

using CUDA
using CUDA.CUSPARSE
using MarkovModels
using SparseArrays
using LinearAlgebra
using LogExpFunctions
using Torch
using Torch: to_tensor

#######################################################################
# Benchmarking macro

macro benchmark(ex)
    quote
        $(esc(ex))
        local elapsedtime = time_ns()
        val = $(esc(ex))
        elapsedtime = time_ns() - elapsedtime
        val, elapsedtime / 1e9
    end
end

#######################################################################
# This is the "standard" way of implementing the forward-backward
# algorithm (as would be done in python for example).

function forward(A, init, lhs::AbstractMatrix{T}) where T
    N = size(lhs, 2)
    log_α = similar(lhs)
    log_α[:, 1] = lhs[:, 1] + init
    for n in 2:N
        #log_α[:, n] = lhs[:,n] + logsumexp(Aᵀ .+ reshape(log_α[:,n-1], 1, :), dims = 2)
        log_α[:, n] = lhs[:,n] + dropdims(logsumexp(A .+ log_α[:,n-1], dims = 1), dims = 1)
    end
    log_α
end

function backward(Aᵀ, final, lhs::AbstractMatrix{T}) where T
    N = size(lhs, 2)
    log_β = similar(lhs)
    log_β[:, end] = final
    for n in N-1:-1:1
        #log_β[:, n] = logsumexp(Aᵀ .+ log_β[:,n+1] .+ lhs[:,n+1], dims = 2)
        log_β[:, n] = dropdims(logsumexp(Aᵀ .+ log_β[:,n+1] .+ lhs[:,n+1], dims = 1), dims = 1)
    end
    log_β
end

function forward_backward(A, Aᵀ, init, final, lhs)
    log_α = forward(A, init, lhs)
    log_β = backward(Aᵀ, final, lhs)
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

function main(T, SF, B, S, N)
    # Generate some pseudo-(log)likelihood
    lhs = convert(Array{SF, 3}, zeros(T, S, N, B))

    # Build the Finite State Machine (FSM).
    fsm = makefsm(SF, S)

    precision = T == Float64 ? "double" : "single"

    #cfsm = compile(fsm, allocator = zeros)
    #Aᵀ = convert(Matrix{T}, cfsm.Aᵀ)
    #A = convert(Matrix{T}, cfsm.A)
    #init = convert(Vector{T}, cfsm.π)
    #final = convert(Vector{T}, cfsm.ω)
    #lhs_rs = convert(Matrix{T}, lhs)
    #val, etime = @benchmark forward_backward(A, Aᵀ, init, final, lhs_rs)
    #println("julia\t$precision\t$S\t$N\tstandard\tdense\tcpu\t$etime")

    #cfsm = compile(fsm, allocator = zeros)
    #αβrecursion(cfsm, lhs)
    #val, etime = @benchmark αβrecursion(cfsm, lhs)
    #println("julia\t$precision\t$S\t$N\tsemifield\tdense\tcpu\t$etime")

    cfsm = compile(fsm, allocator = spzeros)
    αβrecursion(cfsm, lhs)
    val, etime = @benchmark αβrecursion(cfsm, lhs)
    println("julia\t$precision\t$B\t$S\t$N\tsemifield\tsparse\tcpu\t$etime")

    #cfsm = compile(fsm, allocator = zeros)
    #Aᵀ = convert(Matrix{T}, cfsm.Aᵀ) |> CuArray
    #A = convert(Matrix{T}, cfsm.A) |> CuArray
    #init = convert(Vector{T}, cfsm.π) |> CuArray
    #final = convert(Vector{T}, cfsm.ω) |> CuArray
    #lhs_rs = convert(Matrix{T}, lhs) |> CuArray
    #forward_backward(A, Aᵀ, init, final, lhs_rs)
    #val, etime = @benchmark [forward_backward(A, Aᵀ, init, final, lhs_rs) for b in 1:B]
    #println("julia\t$precision\t$B\t$S\t$N\tstandard\tdense\tgpu\t$etime")

    #cfsm = compile(fsm, allocator = zeros) |> gpu
    lhs = CuArray(lhs)
    #αβrecursion(cfsm, lhs)
    #val, etime = @benchmark αβrecursion(cfsm, lhs)
    #println("julia\t$precision\t$S\t$N\tsemifield\tdense\tgpu\t$etime")

    cfsm = compile(fsm, allocator = spzeros) |> gpu
    αβrecursion(cfsm, lhs)
    val, etime = @benchmark αβrecursion(cfsm, lhs)
    println("julia\t$precision\t$B\t$S\t$N\tsemifield\tsparse\tgpu\t$etime")
end

const T = Float64
const SF = LogSemifield{T}
const Bs = 10:10
const Ss = 300:300
const Ns = 500:500:10000

for B in Bs
    for S in Ss
        for N in Ns
            main(T, SF, B, S, N)
        end
    end
end
