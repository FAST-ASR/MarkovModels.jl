# SPDX-License-Identifier: MIT

include("sparsematrixcsr.jl")

const SparseAdjOrTrans{K} = Union{Adjoint{K, <:AbstractSparseMatrix{K}},
                                  Transpose{K, <:AbstractSparseMatrix{K}}} where K

const AnySparseMatrix{K} = Union{SparseAdjOrTrans{K},
                                 AbstractSparseMatrix{K}} where K

const IndexRange = Union{UnitRange, Colon}


#include("cusparse.jl")
