# SPDX-License-Identifier: MIT

using CUDA
using CUDA.CUSPARSE
using JLD2
using MarkovModels
using SparseArrays
using LinearAlgebra

function main(fsmfile, T, B, N; warmup)
    # Load the Finite State Machine (FSM).
    fsm = jldopen(fsmfile, "r") do f
       f["fsm"]
    end

    # Number states in the FSM.
    S = length(fsm)

    # Create a batch of the same FSM.
    fsm = union(repeat([fsm], B)...)

    # Number of pdfs associated with the FSM. This is specific to the FSM
    # and should not be changed unless you use your own FSM.
    npdfs = 84

    # Length of the sequences of the batch. Here we assume that all
    # the sequence have the same length.
    lengths = repeat([N], B)

    # Generate some pseudo-(log)likelihood
    lhs = ones(T, npdfs, N, B)

    # Computing precision
    precision = T == Float64 ? "double" : "single"

    # Dummy run to force compilation
    if warmup pdfposteriors(fsm, lhs, lengths) end

    etime = time_ns()
    pdfposteriors(fsm, lhs, lengths)
    etime = (time_ns() - etime) / 1e9
    println("julia\t$precision\t$N\t$S\t$B\tsemifield\tcpu\t$etime")

    # Move the data and the FSM to the GPU
    fsm = fsm |> MarkovModels.gpu
    lhs = CuArray(lhs)

    # Dummy run to force compilation.
    if warmup pdfposteriors(fsm, lhs, lengths) end

    etime = time_ns()
    pdfposteriors(fsm, lhs, lengths)
    etime = (time_ns() - etime) / 1e9
    println("julia\t$precision\t$N\t$S\t$B\tsemifield\tgpu\t$etime")
end

main(
    "denominator_fsm.jld2", # "numerator_fsm.jld2"
    Float32,    # computing precision
    128,        # batch size
    700;        # sequence length
    warmup=true # if true, make a dummy run first to avoid compilation time
)

