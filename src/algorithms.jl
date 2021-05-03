# Copyright - 2020 - Brno University of Technology
# Copyright - 2021 - CNRS
#
# Contact: Lucas Ondel <lucas.ondel@gmail.com>
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
                     A::AbstractMatrix{T}, lhs::AbstractMatrix) where T
    N = size(lhs, 2)
    Aᵀ = transpose(A)
    buffer = similar(α[:,1], T, length(π))
    α[:, 1] = lhs[:, 1] .* π
    for n in 2:N
        # Equivalent to:
        #αₙ = (Aᵀ * α[:,n-1]) .* lhs[:, n]
        αₙ = mul!(buffer, Aᵀ,
                  view(α, :, n-1) .* view(lhs, :, n), one(T), zero(T))
        α[:, n] = αₙ
    end
    α
end

function βrecursion!(β::AbstractMatrix{T}, ω::AbstractVector{T},
                     A::AbstractMatrix{T}, lhs::AbstractMatrix{T}) where T
    N = size(lhs, 2)
    buffer = similar(β[:,1], T, length(ω))
    β[:, end] = ω
    for n in N-1:-1:1
        # Equivalent to:
        #βₙ = A * (β[:, n+1] .* lhs[:, n+1])
        βₙ = mul!(buffer, A,
                  view(β, :,n+1) .* view(lhs, :, n+1), one(T), zero(T))
        β[:, n] = βₙ
    end
    β
end

"""
    αβrecursion(π, ω, A, lhs)

Baum-Welch algorithm where `π` is initial state probabilities,
`ω` is the final states probabilities and `A` is the probability
transition matrix and `lhs` is the per-state and per-frame likelihood.
"""
function αβrecursion(π::AbstractVector{T}, ω::AbstractVector{T},
                     A::AbstractMatrix{T}, lhs::AbstractMatrix{T}) where T
    S, N = length(π), size(lhs, 2)
    α = zeros(T, S, N)
    β = zeros(T, S, N)
    γ = zeros(T, S, N)

    αrecursion!(α, π, A, lhs)
    βrecursion!(β, ω, A, lhs)

    αβ = α .* β
    sums = sum(αβ, dims = 1)
    αβ ./ sums, maximum(sums)
end

function remove_invalid_αpath!(αₙ::AbstractVector{T}, distances, N, n) where T
    I, V = findnz(αₙ)
    for i in I
        if distances[i] > (N - n + 1)
            αₙ[i] = zero(T)
        end
    end
    αₙ
end

function prune_α!(αₙ, distances, N, n, threshold)
    if ! isnothing(distances) remove_invalid_αpath!(αₙ, distances, N, n) end
    maxval = maximum(αₙ)
    SparseArrays.fkeep!(αₙ, (i,v) -> maximum(αₙ)/v ≤ threshold)
end

function prune_β!(βₙ, n, threshold, α)
    I, V = findnz(α[:, n])
    maxval = maximum(βₙ)
    SparseArrays.fkeep!(βₙ, (i,v) -> i ∈ I && maxval/v ≤ threshold)
end

function αrecursion!(α::AbstractMatrix{T}, π::AbstractVector{T},
                     A::AbstractSparseMatrix{T}, lhs::AbstractMatrix{T},
                     prune!::Function) where T
    N = size(lhs, 2)
    Aᵀ = transpose(A)
    α[:, 1] = lhs[:,1] .* π
    for n in 2:N
        # Equivalent to:
        #αₙ = (Aᵀ * α[:,n-1]) .* lhs[:, n]
        αₙ = mul!(similar(α[:,n], T, size(Aᵀ, 1)), Aᵀ,
                  α[:,n-1] .* lhs[:,n], one(T), zero(T))
        prune!(αₙ, n)
        α[:, n] = αₙ
    end
    α
end

function βrecursion!(β::AbstractMatrix{T}, ω::AbstractVector{T},
                     A::AbstractSparseMatrix{T}, lhs::AbstractMatrix{T},
                     prune!::Function) where T
    N = size(lhs, 2)
    β[:, end] = ω
    for n in N-1:-1:1
        # Equivalent to:
        #βₙ = A * (β[:, n+1] .* lhs[:, n+1])
        βₙ = mul!(similar(β[:, n], T, size(A, 2)), A,
                  β[:,n+1] .* lhs[:, n+1], one(T), zero(T))
        prune!(βₙ, n)
        β[:, n] = βₙ
    end
    β
end


function calculate_distances(ω::AbstractSparseVector{T},
                             A::AbstractSparseMatrix{T}) where T
    Aᵀ = transpose(A)
    queue = Set{Tuple{Int,Int}}([(state, 0) for state in findnz(ω)[1]])
    visited = Set{Int}(findnz(ω)[1])
    distances = zeros(Int, length(ω))
    while ! isempty(queue)
        state, dist = pop!(queue)
        for nextstate in findnz(Aᵀ[state,:])[1]
            if nextstate ∉ visited
                push!(queue, (nextstate, dist + 1))
                push!(visited, nextstate)
                distances[nextstate] = dist + 1
            end
        end
    end
    distances
end

"""
    αβrecursion(π, ω, A::AbstractSparseMatrix, lhs; prune! = identity)

Baum-Welch algorithm operating on sparse matrix, where `π` is initial
state probabilities, `ω` is the final states probabilities and `A` is
the probability transition matrix and `lhs` is the per-state and
per-frame likelihood.
"""
function αβrecursion(π::AbstractVector{T}, ω::AbstractVector{T},
                     A::AbstractSparseMatrix{T}, lhs::AbstractMatrix{T};
                     pruning::Number = T(Inf),
                     distances = nothing) where T
    S, N = length(π), size(lhs, 2)
    α = spzeros(T, S, N)
    β = spzeros(T, S, N)
    γ = spzeros(T, S, N)

    αrecursion!(α, π, A, lhs,
                (αₙ, n) -> prune_α!(αₙ, distances, N, n, pruning))
    βrecursion!(β, ω, A, lhs,
                (βₙ, n) -> prune_β!(βₙ, n, pruning, α))

    normalize_spmatrix(α .* β)
end

function normalize_spmatrix(M::AbstractSparseMatrix{T}) where T
    sums = sum(M, dims = 1)
    I, J, V = findnz(M)
    for i in 1:length(V)
        V[i] /= sums[J[i]]
    end
    sparse(I, J, V), maximum(sums)
end

