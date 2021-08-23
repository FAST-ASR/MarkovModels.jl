# SPDX-License-Identifier: MIT

using CUDA, CUDA.CUSPARSE, SparseArrays
import LogExpFunctions: logsumexp
using MarkovModels
using Test

const T = Float32
const ST = LogSemifield{T}

include("test_semirings.jl")
include("test_fsms.jl")
#include("test_algorithms.jl")

