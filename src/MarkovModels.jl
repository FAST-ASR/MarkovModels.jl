# SPDX-License-Identifier: MIT

module MarkovModels

using CUDA
using CUDA.CUSPARSE
using LinearAlgebra
using SparseArrays

#======================================================================
Redefinition of some linear operations to work on GPU.
======================================================================#

include("linalg.jl")
include("culinalg.jl")

#======================================================================
Semifield algebras.
======================================================================#

include("semifields.jl")

export LogSemifield
export TropicalSemifield

#======================================================================
API to build and manipulate FSM.
======================================================================#

include("fsm.jl")

export FSM
export addstate!
export determinize
export link!
export minimize
export renormalize!
export setinit!
export setfinal!
export states
export remove_eps

include("cfsm.jl")

export CompiledFSM
export compile
export gpu

#======================================================================
Inference algorithms.
======================================================================#

include("algorithms.jl")

export stateposteriors
export bestpath

end
