### A Pluto.jl notebook ###
# v0.18.4

using Markdown
using InteractiveUtils

# ╔═╡ 6e84e042-b5a5-11ec-3bda-e7f5e804b97e
begin
	using Pkg
	Pkg.activate("../")

	using Revise
	using MarkovModels
	using Semirings
	using SparseArrays
end

# ╔═╡ 341f660d-d1be-4fa7-9d11-763da2181941
Label(x) = UnionConcatSemiring(Set([SymbolSequence([x])]))

# ╔═╡ 238148e8-dfd8-454f-abb7-9571d09958a6
K = ProbSemiring{Float32}

# ╔═╡ 488a48c2-da35-4ec0-845f-48b4b8dd3df2
zero(K)

# ╔═╡ 2d2b35bb-ce71-4c5a-b5f5-1ea5d10d03e6
SymbolSequence("bas") >= SymbolSequence("ab") 

# ╔═╡ fde24020-6d43-487d-a719-857659a9e655
sparsevec([1], [1])

# ╔═╡ eb27ffd7-2496-4edb-a2b0-e5828278410c
fsm = FSM{K}(
	[1, 0, 0],
	[0.5 0.5 0.0;
	  0.0 0.5 0.5;
	  0.0 0.0 0.0;],
	[0, 0, 1],
	[Label(:a), Label(:b), Label(:c)]
)

# ╔═╡ 0409195e-6a03-4544-a4a9-c9ad2e1d5051
typeof(collect(fsm.λ[1].val)[1])

# ╔═╡ 88a72285-8521-4e5b-9b5c-f76b6ccd207f
v = fsm.α

# ╔═╡ b0fff691-e5dc-4947-8571-de66e791026f
fsm.T' * (fsm.T' * (fsm.T' * (fsm.T' * (fsm.T' * (fsm.T' * v))))) + fsm.ω

# ╔═╡ 657e4548-c776-4f23-a1b6-128d898fc155
prod(fsm.λ) .+ fsm.λ

# ╔═╡ 90c714f7-9389-4c45-a4a9-8cea264e9b40
(fsm.λ[1] * fsm.λ[2] + fsm.λ[3])

# ╔═╡ 4023155a-3b3d-4fb8-9b62-5a4772d7306b
fsm.α[3] == zero(K)

# ╔═╡ f883eddd-3469-49ba-8614-6768f11492f5
zero(fsm.α[3])

# ╔═╡ 546f7484-c945-4fa7-bc07-1099bf7b7591
x = ProbSemiring[1.0, 0.0, 0]

# ╔═╡ e9b896e4-46ba-484f-8bff-0c463228b77a
UnionConcatSemiring(Set([SymbolSequence([:a, :b, :c])]))

# ╔═╡ 0fbe3317-3d24-41fc-84ca-028766731083
nonzeros

# ╔═╡ ad3b9922-601c-4ec3-81d2-f91df8b7c42c
join([1, 2, 3])

# ╔═╡ Cell order:
# ╠═6e84e042-b5a5-11ec-3bda-e7f5e804b97e
# ╠═341f660d-d1be-4fa7-9d11-763da2181941
# ╠═0409195e-6a03-4544-a4a9-c9ad2e1d5051
# ╠═238148e8-dfd8-454f-abb7-9571d09958a6
# ╠═488a48c2-da35-4ec0-845f-48b4b8dd3df2
# ╠═2d2b35bb-ce71-4c5a-b5f5-1ea5d10d03e6
# ╠═fde24020-6d43-487d-a719-857659a9e655
# ╠═eb27ffd7-2496-4edb-a2b0-e5828278410c
# ╠═88a72285-8521-4e5b-9b5c-f76b6ccd207f
# ╠═b0fff691-e5dc-4947-8571-de66e791026f
# ╠═657e4548-c776-4f23-a1b6-128d898fc155
# ╠═90c714f7-9389-4c45-a4a9-8cea264e9b40
# ╠═4023155a-3b3d-4fb8-9b62-5a4772d7306b
# ╠═f883eddd-3469-49ba-8614-6768f11492f5
# ╠═546f7484-c945-4fa7-bc07-1099bf7b7591
# ╠═e9b896e4-46ba-484f-8bff-0c463228b77a
# ╠═0fbe3317-3d24-41fc-84ca-028766731083
# ╠═ad3b9922-601c-4ec3-81d2-f91df8b7c42c
