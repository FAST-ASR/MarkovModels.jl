### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ 71810a80-0534-11ed-136b-bbaeb8a00ff0
begin
	using LinearAlgebra
	using SparseArrays
end

# ╔═╡ 3e42c05c-6c51-4879-91b8-f47183f35fe3
struct SparseLowRankMatrix{
		T, 
		TS <: AbstractSparseMatrix{T},
		TU <: AbstractMatrix{T},
		TV<: AbstractMatrix{T}
	} <: AbstractMatrix{T}
    S::TS
	U::TU
	V::TV
end

# ╔═╡ 9a94d506-5de2-4372-98aa-e6de9e9615ab
M = SparseLowRankMatrix(
	sprandn(3, 3, 0.2),
	randn(3, 2),
	randn(3, 2),
)

# ╔═╡ 6c72fd89-8d49-4dd2-a43c-fc267c0dfe6e
Base.size(M::SparseLowRankMatrix) = size(M.S)

# ╔═╡ 2d5c5c78-162b-4ff9-a75b-8ec8e20d6831
Base.getindex(M::SparseLowRankMatrix, i::Int, j::Int) =
	M.S[i, j] + dot(M.U[i, :], M.V[j, :])

# ╔═╡ 7da8cf1a-387f-4af6-b815-9c7cca8f0a35
md"""
# Sparse Low-Rank Matrix

We introduce and implements a Sparse Low-Rank (SLR) matrix. 

A ``d \times d`` matrix ``\mathbf{M}`` is said to be SLR if it can be expressed as the sum of a sparse and a low-rank matrix:

```math
\mathbf{M} = \mathbf{S} + \mathbf{U} \mathbf{V}^\top
```
where ``\mathbf{S}`` is a ``d \times d`` *sparse* matrix and ``\mathbf{U}`` and ``\mathbf{V}`` are ``d \times k`` arbitrary matrices where ``k < d``. The purpose of using a SLR matrix is performance: for large matrix, the computation load can be drastically reduced thanks to the sparse and low-rank factorization. 
"""

# ╔═╡ 5c2ff7ca-6ae0-4e2a-aa03-be216dfd40fd
Base.:*(A::SparseLowRankMatrix, B::AbstractSparseMatrix) = 
	SparseLowRankMatrix(A.S * B, A.U, B' * A.V)

# ╔═╡ 7dcb81a8-1c61-410d-ba19-11d0a0065601
Base.:*(A::AbstractSparseMatrix, B::SparseLowRankMatrix) = 
	SparseLowRankMatrix(A * B.S, A * B.U, B.V)

# ╔═╡ 97be84c6-c6f2-4428-acb1-9c31a019d68e
x = [1, 2, 3]

# ╔═╡ 82ad19e2-d7f7-4069-8a7b-c5b71904a4d9
M * sprandn(3, 3, 0.2)'

# ╔═╡ ec258ee1-b9a8-4018-921a-57a1dd6af85e
sprandn(3, 3, 0.2) * M

# ╔═╡ 0a2c549d-7a31-4a76-8fa7-1f9669b9654d
M

# ╔═╡ 557666d7-d866-4a7c-8dc8-4a4c8c52e746
similar(M)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.2"
manifest_format = "2.0"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
"""

# ╔═╡ Cell order:
# ╟─71810a80-0534-11ed-136b-bbaeb8a00ff0
# ╟─7da8cf1a-387f-4af6-b815-9c7cca8f0a35
# ╠═3e42c05c-6c51-4879-91b8-f47183f35fe3
# ╠═9a94d506-5de2-4372-98aa-e6de9e9615ab
# ╠═6c72fd89-8d49-4dd2-a43c-fc267c0dfe6e
# ╠═2d5c5c78-162b-4ff9-a75b-8ec8e20d6831
# ╠═5c2ff7ca-6ae0-4e2a-aa03-be216dfd40fd
# ╠═7dcb81a8-1c61-410d-ba19-11d0a0065601
# ╠═97be84c6-c6f2-4428-acb1-9c31a019d68e
# ╠═82ad19e2-d7f7-4069-8a7b-c5b71904a4d9
# ╠═ec258ee1-b9a8-4018-921a-57a1dd6af85e
# ╠═0a2c549d-7a31-4a76-8fa7-1f9669b9654d
# ╠═557666d7-d866-4a7c-8dc8-4a4c8c52e746
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
