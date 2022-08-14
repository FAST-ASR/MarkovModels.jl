### A Pluto.jl notebook ###
# v0.19.4

using Markdown
using InteractiveUtils

# ╔═╡ 20d75084-ddc1-11ec-362a-0d63b16271b7
begin
	using Pkg
	Pkg.activate("../")

	using LinearAlgebra
	using Plots
	using Semirings
	using Revise
	using MarkovModels
	using SparseArrays
end

# ╔═╡ 32c63dfa-4939-4705-a910-25b5de836078
K = ProbSemiring{Float32}

# ╔═╡ 4e8152de-a3fb-4f29-bced-8ab285d0446c
utt = FSM(
	[1 => K(0.7), 2 => K(0.3)],
	[(1, 2) => one(K), (2, 3) => one(K), (2, 4) => one(K), (3, 4) => one(K),
	 (4, 5) => K(0.7)],
	[4 => K(0.3), 5 => one(K)],
	[Label("sil"), Label("i"), Label("sil"), Label("live"), Label("sil")]
) |> renorm

# ╔═╡ d038a5e1-828d-49a1-8d63-33b0f2c3e86f
lexicon = Dict(
	Label("sil") => FSM([1 => one(K)], [], [1 => one(K)], [Label("sil")]),
	Label("i") => FSM([1 => one(K)], [], [1 => one(K)], [Label("aI")]),
	Label("live") => FSM(
		[1 => one(K)],
		[(1, 2) => one(K), (1, 3) => K(0.6), (2, 4) => K(0.4), (3, 4) => one(K),
		 (4, 5) => one(K)], 
		[5 => one(K)], 
		[Label("l"), Label("i"), Label("aI"), Label("v"), Label("e")]) |> renorm,
	Label("hi") => FSM(
		[1 => one(K)],
		[(1, 2) => one(K)], 
		[2 => one(K)], 
		[Label("h"), Label("i")]) |> renorm,
)

# ╔═╡ e15301db-ad13-4032-917c-73bea50b18e2
fsm = utt ∘ lexicon

# ╔═╡ 97bbbc67-9118-4ca7-a2c6-17416c651846
ngrams = totalngramsum(fsm, order = 3)

# ╔═╡ 308d6804-f4b1-476b-aa5c-a707380bb3ac
LanguageModelFSM(ngrams)

# ╔═╡ Cell order:
# ╠═20d75084-ddc1-11ec-362a-0d63b16271b7
# ╠═32c63dfa-4939-4705-a910-25b5de836078
# ╠═4e8152de-a3fb-4f29-bced-8ab285d0446c
# ╠═d038a5e1-828d-49a1-8d63-33b0f2c3e86f
# ╠═e15301db-ad13-4032-917c-73bea50b18e2
# ╠═97bbbc67-9118-4ca7-a2c6-17416c651846
# ╠═308d6804-f4b1-476b-aa5c-a707380bb3ac
