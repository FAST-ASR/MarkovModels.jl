# SPDX-License-Identifier: MIT


function _expand(lhs::AbstractMatrix{K}, seqlength = size(lhs, 2)) where K
    S, N = size(lhs)
    retval = hcat(vcat(lhs, zeros(K, 1, N)), zeros(K, S+1))
    @views fill!(retval[1:end-1, seqlength+1:end], zero(K))
    @views fill!(retval[end, seqlength+1:end], one(K))
    retval
end

function αrecursion(α̂::AbstractVector{K}, T̂ᵀ::AbstractMatrix{K},
                    lhs::AbstractMatrix{K}) where K
    S, N = length(α̂), size(lhs, 2)
    A = similar(lhs, K, S, N)
    buffer = similar(lhs[:, 1])

    @views broadcast!(*, A[:, 1], α̂, lhs[:, 1])
    @views for n in 2:N
        mul!(buffer, T̂ᵀ, A[:, n - 1])
        broadcast!(*, A[:, n], buffer, lhs[:, n])
    end
    A
end

function βrecursion(T̂::AbstractMatrix{K}, lhs::AbstractMatrix{K}) where K
    S, N = size(T̂, 1), size(lhs, 2)
    B = similar(lhs, K, S, N)
    buffer = similar(lhs[:, 1], S)

    @views fill!(B[:, end], one(K))
    @views for n in N-1:-1:1
        broadcast!(*, buffer, B[:, n+1], lhs[:, n+1])
        mul!(B[:, n], T̂, buffer)
    end
    B
end
