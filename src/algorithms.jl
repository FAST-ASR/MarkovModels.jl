# Copyright - 2020 - Brno University of Technology
# Copyright - 2021 - CNRS
#
# Contact: Lucas Ondel <lucas.ondel@gmail.com
#
# legal entity when the software has been created under wage-earning status
# adding underneath, if so required :" contributor(s) : [name of the
# individuals] ([date of creation])
#
# [e-mail of the author(s)]
#
# This software is a computer program whose purpose is to [describe
# functionalities and technical features of your software].
#
# This software is governed by the CeCILL  license under French law and
# abiding by the rules of distribution of free software.  You can  use,
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
#
# As a counterpart to the access to the source code and  rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author,  the holder of the
# economic rights,  and the successive licensors  have only  limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading,  using,  modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean  that it is complicated to manipulate,  and  that  also
# therefore means  that it is reserved for developers  and  experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and,  more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.

function αrecursion!(α::AbstractMatrix{T}, π::AbstractVector{T},
                     A::AbstractMatrix{T}, lhs::AbstractMatrix;
                     prune!::Function = identity) where T
    N = size(lhs, 2)
    Aᵀ = transpose(A)
    α[:, 1] = lhs[:, 1] .* π

    buffer = similar(α[:,1], T, length(π))

    for n in 2:N
        # Equivalent to:
        #αₙ = (Aᵀ * α[:,n-1]) .* lhs[:, n]
        αₙ = mul!(buffer, Aᵀ,
                  view(α, :, n-1) .* view(lhs, :, n), one(T), zero(T))

        #prune!(αₙ, n)
        α[:, n] = αₙ
    end
    α
end

function αrecursion!(α::AbstractMatrix{T}, π::AbstractVector{T},
                     A::AbstractMatrix{T}, lhs::AbstractMatrix;
                     prune!::Function = identity) where T
    N = size(lhs, 2)
    Aᵀ = transpose(A)
    α[:, 1] = lhs[:, 1] .* π

    buffer = similar(α[:,1], T, length(π))

    for n in 2:N
        # Equivalent to:
        #αₙ = (Aᵀ * α[:,n-1]) .* lhs[:, n]
        αₙ = mul!(buffer, Aᵀ,
                  view(α, :, n-1) .* view(lhs, :, n), one(T), zero(T))

        #prune!(αₙ, n)
        α[:, n] = αₙ
    end
    α
end

function βrecursion!(β::AbstractMatrix{T}, ω::AbstractVector{T},
                     A::AbstractMatrix{T}, lhs::AbstractMatrix;
                     prune!::Function = identity) where T
    N = size(lhs, 2)
    β[:, end] = ω

    buffer = similar(β[:,1], T, length(ω))

    for n in N-1:-1:1
        # Equivalent to:
        #βₙ = A * (β[:, n+1] .* lhs[:, n+1])
        βₙ = mul!(buffer, A,
                  view(β, :,n+1) .* view(lhs, :, n+1), one(T), zero(T))

        #prune!(βₙ, n)
        β[:, n] = βₙ
    end
    β
end

function remove_invalid_αpath!(αₙ::AbstractVector{T}, distances, N, n) where T
    for (i,v) in zip(findnz(αₙ)...)
        if distances[i] > (N - n + 1)
            αₙ[i] = zero(T)
        end
    end
    αₙ
end

function prune_α!(αₙ, distances, N, n, threshold)
    #remove_invalid_αpath!(αₙ, distances, N, n)
    SparseArrays.fkeep!(αₙ, (i,v) -> maximum(αₙ)/v ≤ threshold)
end

function prune_β!(βₙ, n, threshold, α)
    I, V = findnz(α[:, n])
    SparseArrays.fkeep!(βₙ, (i,v) -> i ∈ I && maximum(βₙ)/v ≤ threshold)
end


"""
    αβrecursion(pinit, pend, A, lhs[, pruning = nopruning])

Baum-Welch algorithm where `pinit` is initial state probabilities,
`pfinal` is the final states probabilities and `A` is the transition
matrix and `lh` is the per-state and per-frame likelihood.
"""
function αβrecursion(π::AbstractVector{T},
                     ω::AbstractVector{T},
                     A::AbstractMatrix{T},
                     lhs::AbstractMatrix{T};
                     pruning::T = T(Inf),#upperbound(T),
                    ) where T
    S, N = length(π), size(lhs, 2)
    α = zeros(T, S, N)
    β = zeros(T, S, N)
    γ = zeros(T, S, N)

    αrecursion!(α, π, A, lhs,
                prune! = (αₙ, n) -> prune_α!(αₙ, cfsm.distances, N, n, pruning))
    βrecursion!(β, ω, A, lhs,
                prune! = (βₙ, n) -> prune_β!(βₙ, n, pruning, α))

    αβ = α .* β
    αβ ./ sum(αβ, dims = 1)
end
