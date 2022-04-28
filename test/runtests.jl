# SPDX-License-Identifier: MIT

using MarkovModels
using Semirings
using SparseArrays
using Test

@testset verbose=true "FSMs" begin
    include("test_fsms.jl")
end
#
#@testset verbose=true "algorithms" begin
#    include("test_algorithms.jl")
#end
#
#@testset verbose=true "Linear Algebra" begin
#    include("test_linalg.jl")
#end
