# SPDX-License-Identifier: MIT

using CUDA, CUDA.CUSPARSE, SparseArrays
import LogExpFunctions: logsumexp, logaddexp
using MarkovModels
using Test

@testset verbose = true "Semirings" begin
    include("test_semirings.jl")
end

@testset verbose = true "FSMs" begin
    include("test_fsms.jl")
end

@testset verbose = true "algorithms" begin
    include("test_algorithms.jl")
end

@testset verbose = true "Linear Algebra" begin
    include("test_linalg.jl")
end
