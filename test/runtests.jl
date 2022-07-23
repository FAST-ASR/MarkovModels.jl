# SPDX-License-Identifier: MIT

using Adapt
using CUDA, CUDA.CUSPARSE
import LogExpFunctions: logsumexp, logaddexp
import Semirings: val
using LinearAlgebra
using MarkovModels
using Semirings
using SparseArrays
using Test

@testset verbose=true "FSAs" begin
    include("test_fsas.jl")
end

if CUDA.functional()
    @testset verbose=true "linear algebra" begin
        include("test_linalg.jl")
    end
else
    @warn "CUDA is not functional - skipping tests."
end

#@testset verbose=true "algorithms" begin
#    include("test_algorithms.jl")
#end

