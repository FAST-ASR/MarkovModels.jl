# SPDX-License-Identifier: MIT

module  Inference

export pdfposteriors
export maxstateposteriors
export bestpath

using LinearAlgebra
using SparseArrays
using CUDA
using CUDA.CUSPARSE

using ..Semirings
using ..FSMs

include("array.jl")
include("linalg.jl")
include("culinalg.jl")
include("algorithms.jl")

end
