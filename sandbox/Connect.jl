### A Pluto.jl notebook ###
# v0.19.11

using Markdown
using InteractiveUtils

# ╔═╡ b79aee54-195d-11ed-389d-ff3b4ae31b25
begin
	using Pkg
	Pkg.activate("../")

	using Revise
	using Semirings
	using MarkovModels
end

# ╔═╡ 0b25b4b1-2197-49e0-9e52-387949d5a106
K = ProbSemiring{Float32}

# ╔═╡ d1781c3c-af8d-4e2c-b830-d9a08abef91d
fsa1 = FSA(
	K[1, 0, 0],
	K[0  1  0; 0 0 1; 1 0 0],
	K[0, 0, 1],
	["a", "b", "c"]
)

# ╔═╡ 0d526212-eba1-4554-8788-2cb13249df42
fsa2 = FSA(
	K[1, 0, 0, 0],
	K[0 1 0 1; 0 0 1 0; 1 0 0 0; 0 0 0 0],
	K[0, 0, 1, 0],
	["a", "b", "c", "d"]
)

# ╔═╡ 6c6c667d-a128-4709-8d72-2f3147a96e0f


# ╔═╡ Cell order:
# ╠═b79aee54-195d-11ed-389d-ff3b4ae31b25
# ╠═0b25b4b1-2197-49e0-9e52-387949d5a106
# ╠═d1781c3c-af8d-4e2c-b830-d9a08abef91d
# ╠═0d526212-eba1-4554-8788-2cb13249df42
# ╠═6c6c667d-a128-4709-8d72-2f3147a96e0f
