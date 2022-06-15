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

export
    # FSM creation
    FSM,
    Label,
    nstates,

    # FSM operations
    compose,
    determinize,
    minimize,
    propagate,
    renorm,

    # Total sum algorithm and its variants
    totalcumsum,
    totalsum,
    totalweightsum,
    totallabelsum,

    # Building n-gram language model
    totalngramsum,
    LanguageModelFSM

include("utils.jl")
include("fsm.jl")
include("fsmops.jl")
include("algorithms.jl")
include("lmfsm.jl")

#export pdfposteriors
#export maxstateposteriors
#export bestpath

end
