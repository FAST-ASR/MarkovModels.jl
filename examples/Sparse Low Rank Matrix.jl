### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ 4d1de7ea-fd1d-11ec-2f17-8502f2049810
begin
	using Pkg
	Pkg.activate("../")

	using LinearAlgebra
	using SparseArrays
end

# ╔═╡ ca3c1eae-6301-46b4-adcd-31b80c3c09fb
begin
	struct SparseLowRank{T, TS <: AbstractSparseMatrix{T},
						Tus <: Tuple{Vararg{AbstractSparseVector{T}}},
						Tvs <: Tuple{Vararg{AbstractSparseVector{T}}}} <: AbstractMatrix{T} 
		S::TS
		us::Tus
		vs::Tvs
	end

	Base.size(M::SparseLowRank, i) = size(M.S, i) 
	Base.size(M::SparseLowRank) = size(M.S) 
end

# ╔═╡ d570b0a3-eab3-4cd5-b351-72287c8c4745
S = sprandn(3, 3, 0.3)

# ╔═╡ 9f1af18d-54a0-4cdd-ac4a-a095b3f5bc66
u1 = sprandn(3, 0.3)

# ╔═╡ 9c56d633-de9c-4439-a3c7-4e915524c7e0
u2 = sprandn(3, 0.3)

# ╔═╡ 3d708247-74da-4a35-9948-c3fa891277ef
v1 = sprandn(3, 0.3)

# ╔═╡ eceed4c8-4673-4973-a2d0-0db2ff7d4a34
v2 = sprandn(3, 0.3)

# ╔═╡ a6880717-320e-4f4f-82cb-1c77f91f4169
M = SparseLowRank(S, (u1, u2), (v1, v2));

# ╔═╡ 73509a60-11ad-464b-98bc-cf12e42325b1
size(M, 2)

# ╔═╡ Cell order:
# ╠═4d1de7ea-fd1d-11ec-2f17-8502f2049810
# ╠═ca3c1eae-6301-46b4-adcd-31b80c3c09fb
# ╠═d570b0a3-eab3-4cd5-b351-72287c8c4745
# ╠═9f1af18d-54a0-4cdd-ac4a-a095b3f5bc66
# ╠═9c56d633-de9c-4439-a3c7-4e915524c7e0
# ╠═3d708247-74da-4a35-9948-c3fa891277ef
# ╠═eceed4c8-4673-4973-a2d0-0db2ff7d4a34
# ╠═a6880717-320e-4f4f-82cb-1c77f91f4169
# ╠═73509a60-11ad-464b-98bc-cf12e42325b1
