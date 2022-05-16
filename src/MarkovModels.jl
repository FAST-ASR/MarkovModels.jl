# SPDX-License-Identifier: MIT

module MarkovModels

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
export totalweightsum, totallabelsum

include("fsmops.jl")

##======================================================================
#API to build and manipulate FSM.
#======================================================================#
#
#include("fsms/FSMs.jl")
#
#using .FSMs
#
#export AbstractFSM
#export states
#export arcs
#export semiring
#
#export AbstractMutableFSM
#export addstate!
#export addarc!
#
#export State
#export isinit
#export isfinal
#
#export Arc
#
#export VectorFSM
#export HierarchicalFSM
#export MatrixFSM
#export UnionMatrixFSM
#
#export gpu
#export determinize
#export minimize
#export remove_label
#export renormalize
#export transpose
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
