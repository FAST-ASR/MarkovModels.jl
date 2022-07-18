### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ 71810a80-0534-11ed-136b-bbaeb8a00ff0
begin
	using Pkg
	Pkg.activate("../")
	
	using LinearAlgebra
	using SparseArrays
	using Semirings
	
	using Revise
	using MarkovModels
end

# ╔═╡ 7da8cf1a-387f-4af6-b815-9c7cca8f0a35
md"""
# Composition

Implementation of the composition algorithm
"""

# ╔═╡ 3519d7b9-7b3c-4b04-b9a9-fe49bbae9ec5
K = LogSemiring{Float64}

# ╔═╡ 3e42c05c-6c51-4879-91b8-f47183f35fe3
A = FSM(
	[1 => one(K), 2 => one(K)],
	[(1, 3) => K(1), (3, 3) => K(1)],
	[2 => K(2.5), 3 => K(0)],
	[Label("b"), Label("c"), Label("d")]
)

# ╔═╡ 9a94d506-5de2-4372-98aa-e6de9e9615ab
B = FSM(
	[1 => one(K), 2 => one(K)],
	[(1, 3) => K(1), (3, 4) => K(2.5), (2, 4) => K(3), (4, 4) => K(1.5)],
	[4 => K(2)],
	[Label("b"), Label("d"), Label("c"), Label("d")]
)

# ╔═╡ Cell order:
# ╠═71810a80-0534-11ed-136b-bbaeb8a00ff0
# ╟─7da8cf1a-387f-4af6-b815-9c7cca8f0a35
# ╠═3519d7b9-7b3c-4b04-b9a9-fe49bbae9ec5
# ╠═3e42c05c-6c51-4879-91b8-f47183f35fe3
# ╠═9a94d506-5de2-4372-98aa-e6de9e9615ab
