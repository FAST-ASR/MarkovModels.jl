# SPDX-License-Identifier: MIT

using CUDA
using CUDA.CUSPARSE
using HDF5
using JLD2
using MarkovModels
using SparseArrays
using LinearAlgebra

function makefsm(T, S, B)
    #fsm = FSM{LogSemifield{T}}()

    #prev = addstate!(fsm, pdfindex = 1)
    #setinit!(prev)
    #addarc!(fsm, prev, prev)
    #for s in 2:S
    #    initweight =
    #    state = addstate!(fsm, s; initweight )
    #    addarc!(fsm, prev, state)
    #    addarc!(fsm, state, state)
    #    prev = state
    #end
    #setfinal!(prev)
    #renormalize!(fsm)
    #fsm

   #fsm = jldopen("/people/ondel/Repositories/SpeechLab/recipes/lfmmi/graphs/wsj/train_alignments_fsms.jld2", "r") do f
   #    f["48ic030f"]
   #end
    fsm = jldopen("/people/ondel/Repositories/SpeechLab/recipes/lfmmi/graphs/wsj/denominator_fsm.jld2", "r") do f
       f["fsm"]
    end
    union(repeat([fsm], B)...)
end

function main(T, B, S, N; warmup)
    # Build the Finite State Machine (FSM).
    fsm = makefsm(T, S, B)

    S = length(fsm)

    lengths = repeat([N], B)
    println(lengths)

    # Generate some pseudo-(log)likelihood
    lhs = ones(T, 84, N, B)

    println(size(lhs))

    precision = T == Float64 ? "double" : "single"

    if warmup pdfposteriors(fsm, lhs, lengths) end
    etime = time_ns()
    pdfposteriors(fsm, lhs, lengths)
    etime = (time_ns() - etime) / 1e9
    println("julia\t$precision\t$B\t$S\t$N\tsemifield\tsparse\tcpu\t$etime")

    fsm = fsm |> MarkovModels.gpu
    lhs = CuArray(lhs)
    if warmup pdfposteriors(fsm, lhs, lengths) end
    etime = time_ns()
    pdfposteriors(fsm, lhs, lengths)
    etime = (time_ns() - etime) / 1e9
    println("julia\t$precision\t$B\t$S\t$N\tsemifield\tsparse\tgpu\t$etime")
end

const T = Float32
const Bs = 10:10
const Ss = 300:300:1500
const Ns = 500:500:10000

warmup = true
main(T, 128, 1, 700; warmup=true)

