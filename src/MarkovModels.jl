# SPDX-License-Identifier: MIT

module MarkovModels

using CUDA, CUDA.CUSPARSE, LinearAlgebra, SparseArrays
using LogExpFunctions: logaddexp
using StatsBase: sample, Weights

export FSM, LogSemifield
export αrecursion, βrecursion, αβrecursion
export addstate!, compile, determinize, gpu, link!, minimize,
    renormalize!, setinit!, setfinal!, states

include("cusparse.jl")
include("semifields.jl")
include("fsm.jl")
include("algorithms.jl")

end
