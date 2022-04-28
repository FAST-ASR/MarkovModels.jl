# SPDX-License-Identifier: MIT

# Convert a semiring array `a` to another semiring binary array `b`,
# i.e. `b[i] = one(K)` if `! iszero(a[i])`.
function tobinary(K::Type{<:Semiring}, a::AbstractArray{<:Semiring})
    b = fill!(similar(a, K), zero(K))
    for i in 1:length(a)
        if ! iszero(a[i])
            b[i] = one(K)
        end
    end
    b
end
