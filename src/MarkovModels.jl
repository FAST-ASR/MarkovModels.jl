# SPDX-License-Identifier: MIT

module MarkovModels

using Adapt
using Base.Broadcast: BroadcastStyle, Broadcasted, flatten
using CUDA
using CUDA.CUSPARSE
using LinearAlgebra
using Markdown
using Printf: @sprintf
using SparseArrays
using Semirings

export
    # FSA
    FSA,
    FactorizedFSA,
    SparseFSA,
    DenseForwardFSA,
    DenseFSA,

    # FSA properties
    Properties,
    accessible,
    acyclic,
    coaccessible,
    deterministic,
    lexsorted,
    normalized,
    topsorted,
    weightpropagated,

    # Iterators
    FSAIterator,
    AcyclicFSAIterator,

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


include("linalg/linalg.jl")
#include("fsa/fsa.jl")

#include("fsa_ops.jl")
#include("lmfsa.jl")
#include("inference.jl")


#export maxstateposteriors
#export bestpath

end
