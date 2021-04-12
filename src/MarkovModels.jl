module MarkovModels

using LinearAlgebra
using SparseArrays
using SemifieldAlgebra
using StatsBase: sample, Weights
using StatsFuns: logaddexp

export LogSemifield
export TropicalSemifield

include("semifields.jl")

#######################################################################
# Algorithms for inference with Markov chains

export αrecursion
export αβrecursion
export βrecursion
export resps
export beststring
export samplestring

include("algorithms.jl")

end

