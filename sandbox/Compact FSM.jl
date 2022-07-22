### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ 47e5098e-0547-11ed-3bcb-b5efb72654e8
begin
	using Pkg
	Pkg.activate("../")

	using Revise
	using LinearAlgebra
	using MarkovModels
	using Semirings
	using SparseArrays
end

# ╔═╡ e6ac8d98-2e76-4cab-bc1d-770c98280d43
K = ProbSemiring{Float64}

# ╔═╡ 8ccaa6d2-9038-48a7-aace-1d9c6df7ff9c
HMM() = FSA(
	[1 => one(K)],
	[
		(1, 2) => one(K),
		(2, 2) => one(K)
	],
	[2 => one(K)],
) 

# ╔═╡ 2bbfd1a0-6fce-4465-837d-de82ec47566f
HMM()

# ╔═╡ c8dd5117-ec53-4a86-82f8-de99eab61464
hmms = Dict(
	"a" => HMM(),
	"b" => HMM(),
	"c" => HMM()
)

# ╔═╡ 191b642f-3d43-4773-8529-f512fc47105f
G = FSA(
	[1 => one(K)],
	[
		(1, 2) => one(K),
		(1, 0) => one(K),
		(2, 0) => one(K),
		(3, -1) => one(K),
		(-1, 1) => K(1/2),
		(-1, 2) => K(1/3),
		(0, 3) => K(1/6),
		(0, -1) => one(K)
	],
	[3 => one(K)],
	[Label("a"), Label("b"), Label("c")]
) 

# ╔═╡ 695dc1e6-962b-4220-9d6d-037183ac0a0f
lmul!(G.T, G.T)

# ╔═╡ e10034d3-e83c-4fc3-8849-50f3890956fd
rmepsilon(G)

# ╔═╡ aa143d9f-9948-4229-a120-bb3e1c6812a4
union(G, G, G)

# ╔═╡ da2567ad-9c34-48a2-8181-ba15d521b831
cat(G, G, G) |> rmepsilon  

# ╔═╡ 571a916c-fd8c-4087-8714-6e004c911458
G

# ╔═╡ de827109-e516-4161-a1b2-7f6e3bf03e86
Z = one(K) ./ (sum(G.T, dims=2) .+ G.ω)

# ╔═╡ 7be578ba-1546-4b1c-baf3-6ab89a511c05
G |> rmepsilon |> renorm

# ╔═╡ b05848ed-7ec5-49be-a56f-22f697e82455
replace(G) do i
	hmms[val(G.λ[i])[1]]
end

# ╔═╡ 0eed9252-3722-4cdf-90b4-83d8617005a0
typeof(G.λ[1]), eltype(hmms["a"].λ)

# ╔═╡ d2359e3d-62ae-4a84-85db-efac6eda5f86
hmms["a"].λ[1:2]

# ╔═╡ 0c7d0d5a-92ef-47d3-8385-4572b65ed676
typeof(G)

# ╔═╡ 3b9a3b9f-9896-4952-a31e-0ab7c98e9b36
spdiagm(G.α) * G.T

# ╔═╡ 9ae1bf4d-c08e-4b3f-a2ae-de239a7d132d
hasepsilons(G |> rmepsilon)

# ╔═╡ d05e202b-9493-4c8a-9139-71d87f094ada
begin
	struct SquaresVector <: AbstractArray{Int, 1}
           count::Int
       end
	Base.size(S::SquaresVector) = (S.count,)
	Base.IndexStyle(::Type{<:SquaresVector}) = IndexLinear()
	Base.getindex(S::SquaresVector, i::Int) = i*i

	sqv = SquaresVector(4)
	sqv[2:3], sqv
end

# ╔═╡ a5e2e02d-1f19-4577-9a30-af5144a2cf6d
size(G.T.S), size(G.T.U), size(G.T.D), size(G.T.V)

# ╔═╡ bfc6906f-f37b-49d6-979d-59460b5b47bd
collect([Label(1), Label(2), Label(3)])

# ╔═╡ d2663a8c-fef0-4ec4-a425-6db4f050e038
G.T.S + G.T.U * (I + G.T.D) * G.T.V'

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
# ╠═2bbfd1a0-6fce-4465-837d-de82ec47566f
# ╠═c8dd5117-ec53-4a86-82f8-de99eab61464
# ╠═191b642f-3d43-4773-8529-f512fc47105f
# ╠═a5e2e02d-1f19-4577-9a30-af5144a2cf6d
# ╠═695dc1e6-962b-4220-9d6d-037183ac0a0f
# ╠═e10034d3-e83c-4fc3-8849-50f3890956fd
# ╠═aa143d9f-9948-4229-a120-bb3e1c6812a4
# ╠═da2567ad-9c34-48a2-8181-ba15d521b831
# ╠═571a916c-fd8c-4087-8714-6e004c911458
# ╠═de827109-e516-4161-a1b2-7f6e3bf03e86
# ╠═7be578ba-1546-4b1c-baf3-6ab89a511c05
# ╠═b05848ed-7ec5-49be-a56f-22f697e82455
# ╠═0eed9252-3722-4cdf-90b4-83d8617005a0
# ╠═d2359e3d-62ae-4a84-85db-efac6eda5f86
# ╠═0c7d0d5a-92ef-47d3-8385-4572b65ed676
# ╠═3b9a3b9f-9896-4952-a31e-0ab7c98e9b36
# ╠═9ae1bf4d-c08e-4b3f-a2ae-de239a7d132d
# ╠═d05e202b-9493-4c8a-9139-71d87f094ada
# ╠═bfc6906f-f37b-49d6-979d-59460b5b47bd
# ╠═d2663a8c-fef0-4ec4-a425-6db4f050e038
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
