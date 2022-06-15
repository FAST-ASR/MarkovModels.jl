# SPDX-License-Identifier: MIT

module MarkovModels

using Adapt
using CUDA
using CUDA.CUSPARSE
using JSON
using LinearAlgebra
using SparseArrays
using BlockDiagonals
using Semirings

include("utils.jl")

export FSM
export Label
export nstates

include("fsm.jl")

export compose
export determinize
export minimize
export propagate
export renorm

include("fsmops.jl")

export totcalcumsum
export totalsum
export totalweightsum
export totallabelsum

include("algorithms.jl")

export totalngramsum
export LanguageModelFSM

include("lmfsm.jl")

#
##======================================================================
#Inference algorithms.
#======================================================================#
#
#include("inference/Inference.jl")
#
#using .Inference
#
#export pdfposteriors
#export maxstateposteriors
#export bestpath

end
