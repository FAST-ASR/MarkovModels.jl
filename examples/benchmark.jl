# SPDX-License-Identifier: MIT

using CUDA
using CUDA.CUSPARSE
using MarkovModels
using SparseArrays
using LinearAlgebra

function makefsm(T, S)
    fsm = FSM{LogSemifield{T}}()

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

function main(T, B, S, N; warmup)
    # Generate some pseudo-(log)likelihood
    lhs = ones(T, S, N, B)

    # Build the Finite State Machine (FSM).
    fsm = makefsm(T, S)

    precision = T == Float64 ? "double" : "single"

    cfsm = compile(fsm, allocator = spzeros)
    if warmup stateposteriors(cfsm, lhs) end
    etime = time_ns()
    stateposteriors(cfsm, lhs)
    etime = (time_ns() - etime) / 1e9
    println("julia\t$precision\t$B\t$S\t$N\tsemifield\tsparse\tcpu\t$etime")

    cfsm = compile(fsm, allocator = spzeros) |> gpu
    lhs = CuArray(lhs)
    if warmup stateposteriors(cfsm, lhs) end
    etime = time_ns()
    stateposteriors(cfsm, lhs)
    etime = (time_ns() - etime) / 1e9
    println("julia\t$precision\t$B\t$S\t$N\tsemifield\tsparse\tgpu\t$etime")
end

const T = Float32
const Bs = 10:10
const Ss = 300:300:1500
const Ns = 500:500:10000

warmup = true
for B in Bs
    for S in Ss
        for N in Ns
            if N < S continue end
            main(T, B, S, N; warmup)
            global warmup = false
        end
    end
end
