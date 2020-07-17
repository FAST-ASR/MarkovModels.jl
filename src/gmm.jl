using Distributions
using LinearAlgebra

########################################################################
# GaussianMixtureModel

export GMM

const GMM{D<:AbstractMvNormal} = AbstractMixtureModel{Multivariate, Continuous, D}

function GMM(μ::AbstractMatrix, Σ::AbstractVector;
        π = vec(repeat([1/size(μ,2)], size(μ,2)))::Vector) where T <: AbstractFloat where N
    d = [MvNormal(μ[:,i], Σ[i]) for i in 1:size(μ,2)]
    MixtureModel(d, π)
end


"""GMM(C::Int, D::Int)

Gaussian Mixture Model with C components and D-dimensional input space.

Each GMM component consists of randomly (Normal distr.) initialize mean vector and isotropic covariance matrix.
C - Number of components
D - Dimensionality of input space
"""
GMM(C::Int, D::Int; σ2 = 1.0, kwargs...) = begin
    μ = randn(D,C)
    Σ = vec([Diagonal(repeat([σ2], D)) for _ in 1:C])
    GMM(μ, Σ; kwargs...)
end


"""GMM(X::AbstractMatrix)

Estimate the mean and variance from data.
C - Number of components
ϵ - 
"""
GMM(C::Int, X::AbstractMatrix; ϵ=1e-1) = begin
    m = mean(X, dims=2)
    μ = vec([m + randn(size(X,1)) .* ϵ for _ in 1:C])
    v = var(X; dims=2)
    Σ = [Diagonal(vec(v)) for _ in 1:C]
    GMM(hcat(μ...), Σ)
end

Base.getindex(gmm::GMM, i) = Pair(component(gmm, i), probs(gmm)[i])
Base.firstindex(gmm::GMM) = 1
Base.lastindex(gmm::GMM) = length(gmm)
Base.setindex!(gmm::GMM{D}, component::Pair{D, T}, idx::Integer) where D <: AbstractMvNormal where T <: AbstractFloat = begin 
    gmm.prior.p[idx] = component.second
    gmm.components[idx] = component.first
end


########################################################################
# GMM Statistics

export GmmStats
export add!
export stats

import Base:+

struct GmmStats{D<:Distribution}
    model::D
    zero::AbstractArray
    first::AbstractArray
    second::AbstractArray
end

GmmStats(model::Distribution) = begin
    D = length(model)
    C = isa(model, AbstractMixtureModel) ? ncomponents(model) : 1
    GmmStats(model, zeros(C), zeros(D,C), zeros(D,C))
end

function +(a::GmmStats{D}, b::GmmStats{D}) where D <: Distribution
    if (a.model != b.model)
        error("Models are different")
    end
    zero = a.zero + b.zero
    first = a.first + b.first
    second = a.second + b.second
    return Statistics{D}(a.model, zero, first, second)
end

function add!(a::GmmStats{D}, b::GmmStats{D}) where D <: Distribution
    if (a.model != b.model)
        error("Models are different")
    end
    a.zero += b.zero
    a.first += b.first
    a.second += b.second
    return a
end

"""stats(gmm::GMM, γ::AbstractVector, X::AbstractMatrix)

Compute statistics used for training GMM model.

gmm - GMM model
γ   - SxN matrix of posteriors (from forward-backward algorithm)
"""
function stats(gmm::GMM, γ::AbstractVector, X::AbstractMatrix)
    lh = lhpg(gmm, X)
    tmp = lh .* γ'
    zero = sum(tmp, dims=2)
    first = X * tmp'
    second = X.^2 * tmp'
    return GmmStats(gmm, zero, first, second)
end


########################################################################
# Log-likelihoods

export llhpg, lhpg, llhpf, lhpf, avllh, avlh

_llhpg(gmm::GMM, X::AbstractMatrix) = begin
    C,T,D = ncomponents(gmm), size(X,2), length(gmm)
    llh = zeros(C,T) # P(C=c| X)
    W = probs(gmm) # Weights
    for c in 1:C
        g = component(gmm, c)
        llh[c,:] = logpdf(g, X) .+ log.(W[c])
    end
    logev = logsumexp(llh, dims=1) # evidence
    llh = llh .- logev
    return llh, logev
end

"""llhpg(gmm::GMM, X::AbstractMatrix)

Computes the loglikelihood per gaussian and per frame.

Returns:
llh - CxN matrix of loglikelihoods
where C is number of components and N is nframes
"""
llhpg(gmm::GMM, X::AbstractMatrix) = _llhpg(gmm, X)[1]

"""lhpg(gmm::GMM, X::AbstractMatrix)

Computes the likelihood per gaussian and per frame.

Returns:
llh - CxN matrix of loglikelihoods
where C is number of components and N is nframes
"""
lhpg(gmm::GMM, X::AbstractMatrix) = exp.(llhpg(gmm, X))

"""llhpf(gmm::GMM, X::AbstractMatrix)

Computes the loglikelihood per frame.

Returns:
llh - Nx1 vector of loglikelihoods
"""
llhpf(gmm::GMM, X::AbstractMatrix) = vec(_llhpg(gmm, X)[2])

"""lhpf(gmm::GMM, X::AbstractMatrix)

Computes the likelihood per frame.

Returns:
llh - Nx1 vector of loglikelihoods
"""
lhpf(gmm::GMM, X::AbstractMatrix) = exp.(llhpf(gmm, X))

"""avllh(gmm::GMM, X::AbstractMatrix)

Computes the total loglikelihood.

Returns:
llh - scalar
"""
avllh(gmm::GMM, X::AbstractMatrix) = sum(llhpf(gmm, X))

"""avlh(gmm::GMM, X::AbstractMatrix)

Computes the total likelihood.

Returns:
llh - scalar
"""
avlh(gmm::GMM, X::AbstractMatrix) = exp(avllh(gmm, X))


##################################################################################
# Distribution fitting

"""fit(gmm::GMM, x::AbstractMatrix)

Single iteration of EM algorithm for training GMM.

Perform MLE of GMM parameters (μ, Σ, π)
"""
Distributions.fit(gmm::GMM, X::AbstractMatrix) = begin
    C,T,D = ncomponents(gmm), size(X,2), length(gmm)
    γ, TLL = logpdf(gmm, X)
    
    γsum = sum(γ, dims=2)
    W = vec(γsum ./ T)
    μ = X * γ' ./ γsum'
    if isa(gmm, GMM{FullNormal})
        error("unimplemented")
    else
        Σ = X.^2 * γ' ./ γsum' - μ.^2
    end
    GMM(μ, Σ; π=W), TLL
end

"""update!(gmm::GMM, statistics::Statistics{GMM})

Single iteration of EM algorithm for training GMM used in emitting state in HMM.

Perform MLE of GMM parameters (μ, Σ, π)
"""
update!(gmm::GMM, stats::GmmStats) = begin
    γsum = stats.zero
    W = vec(γsum ./ sum(γsum)) # TODO: Not sure
    μ = stats.first ./ γsum'
    if isa(gmm, GMM{FullNormal})
        error("unimplemented")
    else
        Σ = stats.second ./ γsum' - μ.^2
    end
    for c in 1:ncomponents(gmm)
        g = MvNormal(μ[:, c], Σ[:, c])
        gmm[c] = Pair(g, W[c])
    end
end
