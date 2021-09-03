# SPDX-License-Identifier: MIT

module FSMs

export AbstractFSM
export states
export arcs

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
export renormalize
export transpose

using LinearAlgebra
using SparseArrays
using CUDA
using CUDA.CUSPARSE

using ..Semirings

include("abstractfsm.jl")
include("vectorfsm.jl")
include("hierarchicalfsm.jl")
include("matrixfsm.jl")
include("fsmop.jl")

end
