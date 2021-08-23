# SPDX-License-Identifier: MIT

module MarkovModels

using CUDA
using CUDA.CUSPARSE
using LinearAlgebra
using SparseArrays

#======================================================================
Redefinition of some array/linear operations to work on GPU.
======================================================================#

#include("array.jl")
#include("linalg.jl")
#include("culinalg.jl")

#======================================================================
Semiring algebras.
======================================================================#

include("semifields.jl")

export ProbabilitySemifield
export LogSemifield
export TropicalSemiring

#======================================================================
API to build and manipulate FSM.
======================================================================#

include("fsms/abstractfsm.jl")
export AbstractFSM
export AbstractMutableFSM
export states
export arcs
export addstate!
export addarc!

export State
export isinit
export isfinal

export Arc

include("fsms/vectorfsm.jl")
export VectorFSM

include("fsms/fsmop.jl")
export determinize
export minimize
export renormalize
export remove_eps

#=
#
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

=#

end
