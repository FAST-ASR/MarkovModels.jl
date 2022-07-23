# SPDX-License-Identifier: MIT

module MarkovModels

using Adapt
using Base.Broadcast: BroadcastStyle, Broadcasted, flatten
using CUDA
using CUDA.CUSPARSE
using JSON
using LinearAlgebra
using SparseArrays
using Semirings

export
    # FSA
    FSA,
    Label,
    SymbolTable,
    nstates,

    # FSA operations
    compose,
    determinize,
    hasepsilons,
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
    LanguageModelFSA,

    # Inference
    compile,
    CompiledFSA,
    expand,
    αrecursion,
    βrecursion,
    pdfposteriors

include("linalg.jl")
include("fsa.jl")
include("fsa_ops.jl")
include("lmfsa.jl")
include("inference.jl")


#export maxstateposteriors
#export bestpath

end
