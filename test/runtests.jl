# SPDX-License-Identifier: MIT

using CUDA, CUDA.CUSPARSE, SparseArrays
import LogExpFunctions: logsumexp
using MarkovModels
using Test

const T = Float32
const ST = LogSemifield{T}

@testset verbose = true "Semirings" begin
    include("test_semirings.jl")
end

@testset verbose = true "FSMs" begin
    include("test_fsms.jl")
end

#include("test_algorithms.jl")

