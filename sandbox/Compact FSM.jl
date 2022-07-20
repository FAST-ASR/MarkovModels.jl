### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ 47e5098e-0547-11ed-3bcb-b5efb72654e8
begin
	using Pkg
	Pkg.activate("../")

	using Revise
	using MarkovModels
	using Semirings
	using SparseArrays
end

# ╔═╡ e6ac8d98-2e76-4cab-bc1d-770c98280d43
K = ProbSemiring{Float64}

# ╔═╡ 8ccaa6d2-9038-48a7-aace-1d9c6df7ff9c
HMM() = FSM(
	[1 => one(K)],
	[
		(1, 2) => one(K),
		(2, 2) => one(K)
	],
	[2 => one(K)],
	[Label(1), Label(2)]
) |> renorm

# ╔═╡ c8dd5117-ec53-4a86-82f8-de99eab61464
hmms = Dict(
	"a" => HMM(),
	"b" => HMM(),
	"c" => HMM()
)

# ╔═╡ 191b642f-3d43-4773-8529-f512fc47105f
G = FSM(
	[1 => one(K)],
	[
		(1, 0) => one(K),
		(2, 0) => one(K),
		(3, -1) => one(K),
		(-1, 1) => K(1/2),
		(-1, 2) => K(1/3),
		(0, 3) => K(1/6),
	],
	[3 => one(K)],
	[Label("a"), Label("b"), Label("c")]
)

# ╔═╡ 51ce18df-2ca0-4d88-89ff-592eff643277
GH = replace(G) do i
	hmms[val(G.λ[i])[1]]
end

# ╔═╡ b05848ed-7ec5-49be-a56f-22f697e82455
GH.T

# ╔═╡ 41d25708-f2af-4ec0-9b12-e6d7d3eefab8
?MarkovModels.SparseLowRankMatrix

# ╔═╡ 90096772-ef50-4886-932d-20046d2d6fdb
0.5 * 0.167

# ╔═╡ c632f809-132c-459d-b442-4f79a8a3d600
nT = GH.T.S + GH.T.U * GH.T.V'

# ╔═╡ 9a389b47-5298-41de-8084-d75ceb1e5852
FSM(
	GH.α,
	nT,
	GH.ω,
	GH.λ
)

# ╔═╡ 398df239-b808-49fd-b7ac-220735ae712e
nT

# ╔═╡ bee268f9-3581-4391-9047-1c15a9eee164
GH.ω

# ╔═╡ c345135f-50db-48f4-a4ce-5347dcec550a
rmepsilon(GH)

# ╔═╡ 962296bf-70db-4c9c-a232-1e9be409e93a
GH.T̂[1:6,1:6][1:2, 1]

# ╔═╡ 7a718d01-b352-40d5-92e8-4c78bd22eb9c
GH.T̂.S[1:6, end]

# ╔═╡ f9100a5e-1314-4ae1-a571-53699b0ab422
nnz((GH.T̂.U * GH.T̂.V')[1:6, end])

# ╔═╡ Cell order:
# ╠═47e5098e-0547-11ed-3bcb-b5efb72654e8
# ╠═e6ac8d98-2e76-4cab-bc1d-770c98280d43
# ╠═8ccaa6d2-9038-48a7-aace-1d9c6df7ff9c
# ╠═c8dd5117-ec53-4a86-82f8-de99eab61464
# ╠═191b642f-3d43-4773-8529-f512fc47105f
# ╠═51ce18df-2ca0-4d88-89ff-592eff643277
# ╠═b05848ed-7ec5-49be-a56f-22f697e82455
# ╠═41d25708-f2af-4ec0-9b12-e6d7d3eefab8
# ╠═90096772-ef50-4886-932d-20046d2d6fdb
# ╠═c632f809-132c-459d-b442-4f79a8a3d600
# ╠═9a389b47-5298-41de-8084-d75ceb1e5852
# ╠═398df239-b808-49fd-b7ac-220735ae712e
# ╠═bee268f9-3581-4391-9047-1c15a9eee164
# ╠═c345135f-50db-48f4-a4ce-5347dcec550a
# ╠═962296bf-70db-4c9c-a232-1e9be409e93a
# ╠═7a718d01-b352-40d5-92e8-4c78bd22eb9c
# ╠═f9100a5e-1314-4ae1-a571-53699b0ab422
