# SPDX-License-Identifier: MIT

"""
    tobinary(K::Type{<:Semiring}, x)

Create `y`, a sparse vector or a sparse matrix of the same size of `x`
where `y[i] = one(K)` if `x[i] ≂̸ zero(K)`.
"""
tobinary(K::Type{<:Semiring}, x::AbstractSparseVector{<:Semiring}) =
    sparsevec(findnz(x)[1], ones(K, nnz(x)), length(x))
tobinary(K::Type{<:Semiring}, x::AbstractSparseMatrix{<:Semiring}) =
    sparse(findnz(x)[1], findnz(x)[2], ones(K, nnz(x)), size(x)...)

"""
    mapping(K::Type{<:Semiring}, x, y[, match::Function])

Create a sparse mapping matrix `M` such that `M[i, j] = one(K)`
if `match(x[i], y[j])` and `zero(K)` otherwise.
"""
function mapping(K::Type{<:Semiring}, x, y, match = ==)
    I, J, V = [], [], K[]
    for i in 1:length(x), j in 1:length(y)
        if match(x[i], y[i])
            push!(I, i)
            push!(J, j)
            push!(V, one(K))
        end
    end
    sparse(I, J, V)
end
