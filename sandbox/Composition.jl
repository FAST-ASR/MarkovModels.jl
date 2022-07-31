### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ aba927a2-1032-11ed-037f-452f342c54c6
begin
	using Pkg
	Pkg.activate("../")

	using Revise
	using MarkovModels
	using Semirings
	using SparseArrays
end

# ╔═╡ 580e697d-8d0d-464c-b73d-5e00228a7317
K = ProbSemiring{Float64}

# ╔═╡ 8ce8ec21-13e0-4bb6-a272-d8778106cc2c
A = SparseFSA(
	[1 => one(K)],
	[(1, 2) => K(4) + one(K), (2, 3) => K(5), (2, 1) => K(4),
	 (1, 3) => one(K)],
	[2 => one(K), 3 => one(K)],
	["a", "b", "c"]
)

# ╔═╡ 6b07528a-40f1-4021-8606-ced802a30ed9
B = SparseFSA(
	[1 => one(K)],
	[(1, 2) => K(3), (2, 3) => one(K), (3, 4) => one(K), (2, 1) => K(2),
	 (1, 5) => K(4), (5, 3) => K(1/2)],
	[4 => one(K)],
	["a", "b", "a", "c", "e"]
)

# ╔═╡ bd534392-97b9-4f4a-b981-8695018defac
intersect(B, A)

# ╔═╡ e1cf5cb6-f3d1-4489-ad13-f571c0a69b5f
x = [1, 2, 3]

# ╔═╡ cee7820b-b120-4f3a-87e6-fafed79e278f
x[1:3]

# ╔═╡ Cell order:
# ╠═aba927a2-1032-11ed-037f-452f342c54c6
# ╠═580e697d-8d0d-464c-b73d-5e00228a7317
# ╠═8ce8ec21-13e0-4bb6-a272-d8778106cc2c
# ╠═6b07528a-40f1-4021-8606-ced802a30ed9
# ╠═bd534392-97b9-4f4a-b981-8695018defac
# ╠═e1cf5cb6-f3d1-4489-ad13-f571c0a69b5f
# ╠═cee7820b-b120-4f3a-87e6-fafed79e278f
