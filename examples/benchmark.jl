# SPDX-License-Identifier: MIT

using ArgParse
using CUDA
using CUDA.CUSPARSE
using BenchmarkTools
using MarkovModels
using SparseArrays

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

function main(T, SF, S, N, pruning)
    # Generate some pseudo-(log)likelihood
    lhs = convert(Matrix{SF}, randn(T, S, N))

    # Build the Finite State Machine (FSM).
    fsm = makefsm(SF, S)

    println("Setup:")
    println("  float type: $T")
    println("  # states: $S")
    println("  # frames: $N")
    println("  # pruning: $pruning")
    println()

    println("αβrecursion with dense CPU arrays:")
    cfsm = compile(fsm, allocator = zeros)
    @btime αβrecursion($cfsm, $lhs, pruning = SF($pruning))
    println("------------------------------------------------")

    println("αβrecursion with sparse CPU arrays:")
    cfsm = compile(fsm, allocator = spzeros)
    @btime αβrecursion($cfsm, $lhs, pruning = SF($pruning))
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
        help = "number of observation frames"
        arg_type = Int
        default = 1000
    "--pruning", "-p"
        help = "pruning threshold"
        arg_type = Float64
        default = Inf
    "--single-precision"
        help = "use single precision"
        action = :store_true
end

args = parse_args(s)

T = args["single-precision"] ? Float32 : Float64
SF = LogSemifield{T}
S = args["num-states"]
N = args["num-frames"]
pruning = args["pruning"]
main(T, SF, S, N, pruning)
