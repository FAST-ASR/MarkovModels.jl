# SPDX-License-Identifier: MIT

module MarkovModels

using CUDA, CUDA.CUSPARSE, LinearAlgebra, SparseArrays
using LogExpFunctions: logaddexp
using StatsBase: sample, Weights

export FSM, LogSemifield
export αrecursion, βrecursion, αβrecursion
export addstate!, compile, determinize, gpu, link!, minimize,
    renormalize!, setinit!, setfinal!, states, transpose

include("cusparse.jl")

# Semifield algebra.
include("semifields.jl")

# API to build and manipulate FSM.
include("fsm.jl")

# FSM compilation to have inference efficient FSM format.
include("cfsm.jl")

# Inference algorithms
include("algorithms.jl")

end
