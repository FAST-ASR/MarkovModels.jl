# SPDX-License-Identifier: MIT

module MarkovModels

using Adapt
using Base.Broadcast: BroadcastStyle, Broadcasted, flatten
using CUDA
using CUDA.CUSPARSE
using LinearAlgebra
using SparseArrays
using Semirings

export
    # FSA
    FSA,
    FactorizedFSA,
    SparseFSA,
    DenseForwardFSA,

    browse,
    nstates,

    # FSA operations
    compose,
    connect,
    determinize,
    hasepsilons,
    minimize,
    propagate,
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
    batch,
    compile,
    CompiledFSA,
    expand,
    αrecursion,
    βrecursion,
    pdfposteriors

include("linalg.jl")

include("statevector.jl")
include("transmatrix.jl")
include("fsa.jl")
include("factorizedfsa.jl")
include("sparsefsa.jl")
#include("symtable.jl")

include("forwardfsa.jl")
#include("fsa_ops.jl")
#include("lmfsa.jl")
#include("inference.jl")


#export maxstateposteriors
#export bestpath

end
