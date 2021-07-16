# SPDX-License-Identifier: MIT

module MarkovModels

using CUDA, CUDA.CUSPARSE, LinearAlgebra, SparseArrays
using StatsBase: sample, Weights

export FSM, LogSemifield
export αrecursion, βrecursion, αβrecursion
export addstate!, compile, determinize, gpu, link!, minimize,
    renormalize!, setinit!, setfinal!, states, transpose

# Redefinition of  some linear operation to work on GPU.
include("linalg.jl")
include("culinalg.jl")

# Semifield algebras.
include("semifields.jl")

# API to build and manipulate FSM.
include("fsm.jl")

# FSM compilation to have inference efficient FSM format.
include("cfsm.jl")

# Inference algorithms
include("algorithms.jl")

end
