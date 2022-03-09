# SPDX-License-Identifier: MIT

module MarkovModels

using CUDA
using CUDA.CUSPARSE
using LinearAlgebra
using SparseArrays

#======================================================================
Semiring algebras.
======================================================================#

include("semirings/Semirings.jl")

using .Semirings

export Semiring
export Semifield

export ProbabilitySemifield
export LogSemifield
export TropicalSemiring

#======================================================================
API to build and manipulate FSM.
======================================================================#

include("fsms/FSMs.jl")

using .FSMs

export AbstractFSM
export states
export arcs
export semiring

export AbstractMutableFSM
export addstate!
export addarc!

export State
export isinit
export isfinal

export Arc

export VectorFSM
export HierarchicalFSM
export MatrixFSM
export UnionMatrixFSM

export gpu
export determinize
export minimize
export remove_label
export renormalize
export transpose

#======================================================================
Inference algorithms.
======================================================================#

include("inference/Inference.jl")

using .Inference

export pdfposteriors
export maxstateposteriors
export bestpath

end
