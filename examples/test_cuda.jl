### A Pluto.jl notebook ###
# v0.19.8

using Markdown
using InteractiveUtils

# ╔═╡ 3b586fda-ebc5-11ec-3816-e54f891871f0
begin
	using Pkg
	Pkg.activate("../")

	using Adapt
	using CUDA
	using CUDA.CUSPARSE
	using LinearAlgebra
	using Plots
	using Semirings
	using Serialization
	using SparseArrays

	using Revise
	using MarkovModels
end

# ╔═╡ 5114ca9c-e03a-4d80-85a8-868d8b398eb8
K = LogSemiring{Float32}

# ╔═╡ f4c8fd1e-6dc0-4cd9-987f-7b4b5158c454
fsm = FSM(
	[1 => one(K)],
	[(1, 1) => one(K), (1, 2) => one(K), (2, 2) => one(K), (2, 3) => one(K), 
	 (3, 3) => one(K)],
	[3 => one(K)],
	[Label(1), Label(2), Label(3)]
) |> renorm

# ╔═╡ 407398d2-adf0-42b0-b55e-0ca88a027c12
C = sparse([1, 2, 3], [1, 2, 3], [one(K), one(K), one(K)], 3, 3)

# ╔═╡ 3aecb46e-0fb9-4549-aeb6-5cba344a0728
cuC = CUDA.adapt(CuArray, C)

# ╔═╡ 2e1bf652-2b02-45e6-8ced-5d3a038bdbf9
lhs = convert(Array{K}, randn(size(C, 2), 100)) 

# ╔═╡ a916e309-0779-4f14-b10c-bb8291df0dca
culhs = CUDA.adapt(CuArray, lhs)

# ╔═╡ 7839af84-a6da-4453-9d82-90e617553571
A = αrecursion(fsm.α̂, fsm.T̂', MarkovModels._expand(C * lhs))

# ╔═╡ 173bec26-fd0b-4569-bd09-d8ce87b5b976
cuA = αrecursion(CUDA.adapt(CuArray, fsm.α̂), CUDA.adapt(CuArray, fsm.T̂)', MarkovModels._expand(cuC * culhs))

# ╔═╡ d77dec45-d7f2-44a8-a94c-039dc8590e93
B = βrecursion(fsm.T̂, MarkovModels._expand(C * lhs))

# ╔═╡ 1707b2c4-ffb3-4456-b3c4-d1ec7b0bc2d1
cuB = βrecursion(CUDA.adapt(CuArray, fsm.T̂), MarkovModels._expand(cuC * culhs))

# ╔═╡ e7460548-be11-44f7-82b9-04b27bb54a00
Ẑ = (A .* B) 

# ╔═╡ c23511de-de76-4b9e-a0e0-8aaa2c80f055
Z = Ẑ ./ sum(Ẑ, dims = 1)

# ╔═╡ aa198482-9b14-4a0c-9054-fa99d5522857
cuẐ = cuA .* cuB

# ╔═╡ 0c174cb4-bf0d-4c1d-a733-b4d5f17cda83
cuZ = cuẐ ./ sum(cuẐ, dims = 1)

# ╔═╡ b8c00b85-b534-4bab-b6b9-e7a179d5771c
plot((exp ∘ val).((Z[1:end-1, 1:end-1])'))

# ╔═╡ 10e6fc39-b196-4e69-af51-3c88c5629784
plot((exp ∘ val).((C' * Z[1:end-1, 1:end-1])'))

# ╔═╡ 00c23824-2403-4f3f-8de8-bb77a440c83c
plot((exp ∘ val).(CUDA.adapt(Array, cuZ[1:end-1, 1:end-1])'))

# ╔═╡ c16f93ee-daae-4921-ad2e-2d1fd1efd652
plot((exp ∘ val).(CUDA.adapt(Array, cuC' * cuZ[1:end-1, 1:end-1])'))

# ╔═╡ Cell order:
# ╠═3b586fda-ebc5-11ec-3816-e54f891871f0
# ╠═5114ca9c-e03a-4d80-85a8-868d8b398eb8
# ╠═f4c8fd1e-6dc0-4cd9-987f-7b4b5158c454
# ╠═407398d2-adf0-42b0-b55e-0ca88a027c12
# ╠═3aecb46e-0fb9-4549-aeb6-5cba344a0728
# ╠═2e1bf652-2b02-45e6-8ced-5d3a038bdbf9
# ╠═a916e309-0779-4f14-b10c-bb8291df0dca
# ╠═7839af84-a6da-4453-9d82-90e617553571
# ╠═173bec26-fd0b-4569-bd09-d8ce87b5b976
# ╠═d77dec45-d7f2-44a8-a94c-039dc8590e93
# ╠═1707b2c4-ffb3-4456-b3c4-d1ec7b0bc2d1
# ╠═e7460548-be11-44f7-82b9-04b27bb54a00
# ╠═c23511de-de76-4b9e-a0e0-8aaa2c80f055
# ╠═aa198482-9b14-4a0c-9054-fa99d5522857
# ╠═0c174cb4-bf0d-4c1d-a733-b4d5f17cda83
# ╠═b8c00b85-b534-4bab-b6b9-e7a179d5771c
# ╠═10e6fc39-b196-4e69-af51-3c88c5629784
# ╠═00c23824-2403-4f3f-8de8-bb77a440c83c
# ╠═c16f93ee-daae-4921-ad2e-2d1fd1efd652
