# SPDX-License-Identifier: MIT

module MarkovModels

using CUDA
using CUDA.CUSPARSE
using LinearAlgebra
using SparseArrays

#======================================================================
Redefinition of some array/linear operations to work on GPU.
======================================================================#

include("array.jl")
include("linalg.jl")
include("culinalg.jl")

#======================================================================
Semiring algebras.
======================================================================#

include("semifields.jl")

export LogSemifield
export TropicalSemiring

#======================================================================
API to build and manipulate FSM.
======================================================================#

include("fsm.jl")

export FSM
export addstate!
export determinize
export addarc!
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

export pdfposteriors
export maxstateposteriors
export bestpath

end
