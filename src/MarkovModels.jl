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
    FSM,
    Label,
    nstates,

    # FSM operations
    compose,
    determinize,
    minimize,
    propagate,
    rawunion,
    renorm,

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
    compile,
    αrecursion,
    βrecursion,
    pdfposteriors,
    CompiledFSM,

    # LBP
    FactorialFSM,
    CompiledFactorialFSM

include("utils.jl")
include("fsm.jl")
include("fsmops.jl")
include("algorithms.jl")
include("lmfsm.jl")
include("linalg.jl")
include("inference.jl")
include("lbp_inference.jl")


#export maxstateposteriors
#export bestpath

end
