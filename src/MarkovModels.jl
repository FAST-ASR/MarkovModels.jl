# SPDX-License-Identifier: MIT

module MarkovModels

using Adapt
using Base.Broadcast: BroadcastStyle, Broadcasted, flatten
using CUDA
using CUDA.CUSPARSE
using JSON
using LinearAlgebra
using SparseArrays
using BlockDiagonals
using Semirings

export
    # FSM creation
    FSA,
    Label,
    nstates,

    # FSM operations
    compose,
    determinize,
    minimize,
    propagate,
    rawunion,
    renorm,
    rmepsilon,

    # Total sum algorithm and its variants
    totalcumsum,
    totalsum,
    totalweightsum,
    totallabelsum,

    # Building n-gram language model
    totalngramsum,
    LanguageModelFSM,

    # Inference
    expand,
    αrecursion,
    βrecursion,
    pdfposteriors

include("linalg.jl")
include("fsa.jl")
#include("fsmops.jl")
#include("lmfsm.jl")
#include("inference.jl")


#export maxstateposteriors
#export bestpath

end
